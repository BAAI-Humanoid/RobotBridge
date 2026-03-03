/*
This is the transition layer for Deployment on Unitree series robot

The main logic for this code is that it serves as the transition layer between the policy(high-level) and the robots(low-level).

                                                Policy (High Level)
                                                       |
                                                Transition Layer (HERE)
                                                       |
                                                Unitree Robot (Low Level)

The code is based on the prevailing LCM to build communication between different parts and support easy transfer between any unitree robots (For example, G1->H1)

You may define the robot-specific params in unitree_sdk2/assets where you can create a new file and define the params as unitree_g1_29dof.hpp

The detailed implementation of communication is:

                                                Policy (High Level)
                                                   LCM||LCM
                                                Transition Layer (HERE)
                                   (Wrapped by LCM)DDS||DDS
                                                Unitree Robot (Low Level)

The left side represents the direction from up to down and the right side is the opposite.
*/


// Standard Content
#include <cmath>
#include <memory>
#include <atomic>
#include <algorithm>
#include <lcm/lcm-cpp.hpp>

// Unitree
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/HandCmd_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include "unitree/common/thread/thread.hpp"

// LCM
#include "pd_tau_targets_lcmt.hpp"
#include "state_estimator_lcmt.hpp"
#include "body_control_data_lcmt.hpp"
#include "rc_command_lcmt.hpp"
#include "dex_command_lcmt.hpp"

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_STATE_TOPIC = "rt/lowstate";
#define TOPIC_SPORT_STATE "rt/odommodestate"
static const std::string DEX_LEFT_CMD_TOPIC = "rt/dex3/left/cmd";
static const std::string DEX_RIGHT_CMD_TOPIC = "rt/dex3/right/cmd";
    
constexpr int DEX_MOTOR_MAX = 7;
const float dex_max_limits_left[DEX_MOTOR_MAX]=  {  1.05f ,  1.05f  , 1.75f ,   0.f   ,  0.f    , 0.f     , 0.f   };
const float dex_min_limits_left[DEX_MOTOR_MAX]=  { -1.05f , -0.742f ,   0.f  , -1.57f , -1.75f , -1.57f  ,-1.75f};
const float dex_max_limits_right[DEX_MOTOR_MAX]= {  1.05f , 0.742f  ,   0.f  ,  1.57f , 1.75f  , 1.57f  , 1.75f};
const float dex_min_limits_right[DEX_MOTOR_MAX]= { -1.05f , -1.05f  , -1.75f ,    0.f  ,  0.f    ,   0.f   ,0.f    };

typedef struct {
    uint8_t id     : 4;
    uint8_t status : 3;
    uint8_t timeout: 1;
} RIS_Mode_t;

#include "assets/remote_controller.hpp" 

/*You may include different robot configs for different usages; All you need to do is to define the relevant variables in assets/[YOUR UNITREE ROBOT].hpp*/
#include "assets/unitree_g1_29dof.hpp"

/*---------------------------Here is the main body of the controller-----------------------------*/
class RobotController {
    private:
        double time_;
        double control_dt_;  // 0.002s-500HZ
        double duration_;    // time for moving to default pose
        PRorAB mode_;        // mode for control ankle
        uint8_t mode_machine_; 

        /*Communication between control interface and low level humanoid robots*/
        unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
        unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_; //Unitree state subscriber and publisher
        unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> odometer_subscriber_;
        unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> dex_left_publisher_;
        unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> dex_right_publisher_;

        /*Communication between control interface and high level policy*/
        lcm::LCM _simpleLCM;
        state_estimator_lcmt body_state_simple = {0};
        body_control_data_lcmt joint_state_simple = {0};
        pd_tau_targets_lcmt joint_command_simple = {0};
        rc_command_lcmt rc_command = {0};

        /*Multi-threads*/
        unitree::common::ThreadPtr highstateWriterThreadPtr, lowcmdWriterThreadPtr, highcmdReceiverThreadPtr, dexWriterThreadPtr;

        /*Indicators*/
        bool _firstRun;
        bool _firstCommandReceived;
        bool _firstLowCmdReceived;
        bool _firstHighCmdReceived;
        bool _firstOdometerMsgReceived;
        bool damping_mode_;
        int damping_log_count_;
        bool last_l2b_pressed_;
        bool last_l2y_pressed_;

        /*Data buffer*/
        unitree_hg::msg::dds_::LowState_ low_state{};
        unitree_hg::msg::dds_::LowCmd_ low_cmd{};
        unitree_go::msg::dds_::SportModeState_ odometer_state{};
        xRockerBtnDataStruct remote_key_data;
        unitree_hg::msg::dds_::HandCmd_ dex_left_cmd{};
        unitree_hg::msg::dds_::HandCmd_ dex_right_cmd{};
        std::atomic<float> dex_left_target{0.0f};   // 0=open, 1=closed
        std::atomic<float> dex_right_target{0.0f};
        std::atomic<int64_t> dex_last_utime{0};
        


    public:
        RobotController(std::string networkInterface): 
            time_(0.0),
            control_dt_(0.002), // 200HZ
            duration_(5.0), //time for moving to default pose
            mode_(PR), // ankle control mode
            mode_machine_(0),
            damping_mode_(false),
            damping_log_count_(0),
            last_l2b_pressed_(false),
            last_l2y_pressed_(false)
        {
            // Init network connection
            unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);

            set_default_state();
    
            /*-------Create Communication between transition layer and the low-level humanoid robots------*/
            // create publisher (transition layer -> robot)
            lowcmd_publisher_.reset(
                new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(HG_CMD_TOPIC));
            lowcmd_publisher_->InitChannel();
            // create writer (which uses publisher) (transition layer -> robot)
            lowcmdWriterThreadPtr = unitree::common::CreateRecurrentThreadEx("dds_write_thread", UT_CPU_ID_NONE, control_dt_*1e6, &RobotController::lowcmdWriter, this);

            // create subscriber (robot -> transition layer)
            lowstate_subscriber_.reset(
                new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(
                    HG_STATE_TOPIC));
            lowstate_subscriber_->InitChannel(
                std::bind(&RobotController::lowstateHandler, this, std::placeholders::_1), 1);

            odometer_subscriber_.reset(
                new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(
                    TOPIC_SPORT_STATE));
            odometer_subscriber_->InitChannel(
                std::bind(&RobotController::OdometerHandler, this, std::placeholders::_1), 1);

            dex_left_publisher_.reset(
                new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_>(DEX_LEFT_CMD_TOPIC));
            dex_right_publisher_.reset(
                new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_>(DEX_RIGHT_CMD_TOPIC));
            dex_left_publisher_->InitChannel();
            dex_right_publisher_->InitChannel();

            /*--------Create Comminication between transition layer and the high-level policy-------*/
            // create lcm subscriber (policy action -> transition layer); Receiver receives high-level signals and hands over to handler for processing.
            _simpleLCM.subscribe("pd_plustau_targets", &RobotController::highcmdHandler, this);
            _simpleLCM.subscribe("dex_command", &RobotController::dexCommandHandler, this);
            highcmdReceiverThreadPtr = unitree::common::CreateRecurrentThreadEx("lcm_recv_thread", UT_CPU_ID_NONE, control_dt_*1e6, &RobotController::highcmdReceiver, this);

            // lcm send thread (transition layer -> policy)
            highstateWriterThreadPtr = unitree::common::CreateRecurrentThreadEx("lcm_send_thread", UT_CPU_ID_NONE, control_dt_*1e6, &RobotController::highstateWriter, this);
            dexWriterThreadPtr = unitree::common::CreateRecurrentThreadEx("dex3_write_thread", UT_CPU_ID_NONE, 20000, &RobotController::dexWriter, this);

            
            _firstRun = true;
            _firstCommandReceived = false;
            _firstLowCmdReceived = false; 
            _firstHighCmdReceived = false;
            _firstOdometerMsgReceived = false;

            dex_left_cmd.motor_cmd().resize(DEX_MOTOR_MAX);
            dex_right_cmd.motor_cmd().resize(DEX_MOTOR_MAX);
        }

        /*Initialization*/
        void set_default_state(){
            for(int i=0; i<NUM_MOTOR; i++){
                joint_command_simple.q_des[i] = default_joint_position[i];
                joint_command_simple.qd_des[i] = 0;
                joint_command_simple.tau_ff[i] = 0;
                joint_command_simple.kp[i] = Kp[i];
                joint_command_simple.kd[i] = Kd[i];

            }
            std::cout << "Default Joint Position Set!" << std::endl; 
        }

        /*-----------Communication with the high-level layer--------------*/
        // High command receive(It will hand over the control signal to handler): Policy -> Transition Layer
        void highcmdReceiver(){
            while(true){
                _simpleLCM.handle();
            }
        }

        void highcmdHandler(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const pd_tau_targets_lcmt *msg){
            (void) rbuf;
            (void) chan;
            joint_command_simple = *msg;

            if (!_firstHighCmdReceived){
                _firstHighCmdReceived = true;
                std::cout<< "Communication built successfully between transition layer and policy!" << std::endl;
            }
        }

        void dexCommandHandler(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const dex_command_lcmt *msg){
            (void) rbuf;
            (void) chan;
            auto clamp01 = [](float v){ return std::min(std::max(v, 0.0f), 1.0f); };
            dex_left_target.store(clamp01(static_cast<float>(msg->left_grip) / 100.0f));
            dex_right_target.store(clamp01(static_cast<float>(msg->right_grip) / 100.0f));
            dex_last_utime.store(msg->utime);
        }
  
        //High state writer: Transition Layer -> Policy; You may only 
        void highstateWriter() {
            for(int i=0; i<NUM_MOTOR; i++){
                joint_state_simple.q[i] = low_state.motor_state()[i].q();
                joint_state_simple.qd[i] = low_state.motor_state()[i].dq();
                joint_state_simple.tau_est[i] = low_state.motor_state()[i].tau_est();
            }
            
            for(int i=0; i<4; i++)
                body_state_simple.quat[i] = low_state.imu_state().quaternion()[i];
            
            for(int i=0; i<3; i++){
                body_state_simple.rpy[i] = low_state.imu_state().rpy()[i];
                body_state_simple.aBody[i] = low_state.imu_state().accelerometer()[i];
                body_state_simple.omegaBody[i] = low_state.imu_state().gyroscope()[i];
                body_state_simple.p[i] = odometer_state.position()[i];
                body_state_simple.vBody[i] = odometer_state.velocity()[i];
            }

            if(mode_machine_ != low_state.mode_machine()){
                if(mode_machine_ == 0)
                    std::cout << "G1 type: " << unsigned(low_state.mode_machine()) << std::endl;
                mode_machine_ = low_state.mode_machine();
            }

            memcpy(&remote_key_data, &low_state.wireless_remote()[0], 40);
            rc_command.left_stick[0] = remote_key_data.lx;
            rc_command.left_stick[1] = remote_key_data.ly;
            rc_command.right_stick[0] = remote_key_data.rx;
            rc_command.right_stick[1] = remote_key_data.ry;
            rc_command.right_lower_right_switch = remote_key_data.btn.components.R2;
            rc_command.right_upper_switch = remote_key_data.btn.components.R1;
            rc_command.left_lower_left_switch = remote_key_data.btn.components.L2;
            rc_command.left_upper_switch = remote_key_data.btn.components.L1;

            _simpleLCM.publish("state_estimator_data", &body_state_simple);
            _simpleLCM.publish("body_control_data", &joint_state_simple);
            _simpleLCM.publish("rc_command_data", &rc_command);
        }


        void lowstateHandler(const void *message) {
            /*
            The lowstateHandler is mainly responsible for the following things:
            1. Update the current proprioception state
            2. Obtain the remote controller state
            3. Update the signal across threads
            */
        
            low_state = *(const unitree_hg::msg::dds_::LowState_ *)message;

            if (low_state.crc() != Crc32Core((uint32_t *)&low_state, (sizeof(unitree_hg::msg::dds_::LowState_) >> 2) - 1)) 
            {
                std::cout << "low_state CRC Error" << std::endl;
                return;
            }

            if (_firstLowCmdReceived == false)
            {
                std::cout << "Communication built successfully between transition layer and robot!" <<std::endl;
                _firstLowCmdReceived = true;
            }
        }

        void OdometerHandler(const void *message) {
            
            odometer_state = *(unitree_go::msg::dds_::SportModeState_ *) message;
            
            if(_firstOdometerMsgReceived == false)
            {
                std::cout << "Commnication built successfully between transition layer and odometer!" << std::endl;
                _firstOdometerMsgReceived = true;
            }
        }

        void fillDexCmd(unitree_hg::msg::dds_::HandCmd_ &cmd, 
                        bool isLeftHand, 
                        float gripRatio,
                        bool dampingMode) {

            gripRatio = std::clamp(gripRatio, 0.0f, 1.0f);
            gripRatio = 1.0f - gripRatio;

            if (cmd.motor_cmd().size() != DEX_MOTOR_MAX) {
                cmd.motor_cmd().resize(DEX_MOTOR_MAX);
            }

            const float* maxLimits = isLeftHand ? dex_max_limits_left : dex_max_limits_right;
            const float* minLimits = isLeftHand ? dex_min_limits_left : dex_min_limits_right;

            for (int i = 0; i < DEX_MOTOR_MAX; i++) {
                RIS_Mode_t ris_mode{};
                ris_mode.id = i;
                ris_mode.status = 0x01;
                ris_mode.timeout = 0x00;

                uint8_t mode = 0;
                mode |= (ris_mode.id & 0x0F);
                mode |= (ris_mode.status & 0x07) << 4;
                mode |= (ris_mode.timeout & 0x01) << 7;

                cmd.motor_cmd()[i].mode(mode);
                cmd.motor_cmd()[i].tau(0);
                cmd.motor_cmd()[i].dq(0);
                cmd.motor_cmd()[i].kp(dampingMode ? 0.0f : 1.2f);
                cmd.motor_cmd()[i].kd(dampingMode ? 0.2f : 0.1f);

                float mid = (maxLimits[i] + minLimits[i]) * 0.5f;

                if (dampingMode) {
                    cmd.motor_cmd()[i].q(mid);
                } 
                else {
                    if (i == 0) {
                        // thumb_0: 旋转关节保持中位
                        cmd.motor_cmd()[i].q(mid);
                    }
                    else if (i == 1) {
                        // 🔧 thumb_1: 左右手方向相反
                        float open_q, close_q;

                        if (isLeftHand) {
                            open_q = minLimits[i];   // -0.742
                            close_q = maxLimits[i];  // 1.05
                        } else {
                            open_q = maxLimits[i];   // 0.742  ← 反向
                            close_q = minLimits[i];  // -1.05  ← 反向
                        }

                        float target_q = open_q + gripRatio * (close_q - open_q);
                        cmd.motor_cmd()[i].q(target_q);
                    }
                    else if (i == 2) {
                        // thumb_2: 从0开始（左右手逻辑相同）
                        float open_q = 0.0f;
                        float close_q = isLeftHand ? maxLimits[i] : minLimits[i];

                        float target_q = open_q + gripRatio * (close_q - open_q);
                        float limit_min = std::min(minLimits[i], maxLimits[i]);
                        float limit_max = std::max(minLimits[i], maxLimits[i]);
                        target_q = std::clamp(target_q, limit_min, limit_max);

                        cmd.motor_cmd()[i].q(target_q);
                    }
                    else {
                        // 中指和食指 (3-6): 从0开始
                        float open_q = 0.0f;
                        float close_q = isLeftHand ? minLimits[i] : maxLimits[i];

                        float target_q = open_q + gripRatio * (close_q - open_q);
                        float limit_min = std::min(minLimits[i], maxLimits[i]);
                        float limit_max = std::max(minLimits[i], maxLimits[i]);
                        target_q = std::clamp(target_q, limit_min, limit_max);

                        cmd.motor_cmd()[i].q(target_q);
                    }
                }
            }
        }
                
        void dexWriter() {
            if (!dex_left_publisher_ && !dex_right_publisher_) {
                return;
            }

            bool handDamping = damping_mode_;
            fillDexCmd(dex_left_cmd, true, dex_left_target.load(), handDamping);
            fillDexCmd(dex_right_cmd, false, dex_right_target.load(), handDamping);

            if (dex_left_publisher_) {
                dex_left_publisher_->Write(dex_left_cmd);
            }
            if (dex_right_publisher_) {
                dex_right_publisher_->Write(dex_right_cmd);
            }
        }

        void lowcmdWriter() {
            
            low_cmd.mode_pr() = mode_;
            low_cmd.mode_machine() = mode_machine_;

            if(time_ < duration_){
                time_ += control_dt_;
                
                float ratio = time_ / duration_;
                for(int i = 0; i<NUM_MOTOR; i++){
                    low_cmd.motor_cmd().at(i).mode() = 1;
                    low_cmd.motor_cmd()[i].kp() = Kp[i];
                    low_cmd.motor_cmd()[i].kd() = Kd[i];
                    low_cmd.motor_cmd()[i].dq() = 0.f;
                    low_cmd.motor_cmd()[i].tau() = 0.f;
                    
                    float q_des = default_joint_position[i];
                    
                    q_des = (q_des - joint_state_simple.q[i]) * ratio + joint_state_simple.q[i];
                    low_cmd.motor_cmd()[i].q() = q_des;
                }
            }

            else{
                if (_firstRun){
                    for(int i=0; i<NUM_MOTOR; i++)
                        joint_command_simple.q_des[i] = joint_state_simple.q[i];
                    remote_key_data.btn.components.Y = 0;
                    remote_key_data.btn.components.A = 0;
                    remote_key_data.btn.components.B = 0;
                    remote_key_data.btn.components.L2 = 0;
                    _firstRun = false;
                }

                bool l2b_pressed = ((int) remote_key_data.btn.components.B ==1 && (int) remote_key_data.btn.components.L2 == 1);
                bool l2y_pressed = ((int) remote_key_data.btn.components.Y ==1 && (int) remote_key_data.btn.components.L2 == 1);
                bool l2b_edge = l2b_pressed && !last_l2b_pressed_;
                bool l2y_edge = l2y_pressed && !last_l2y_pressed_;
                last_l2b_pressed_ = l2b_pressed;
                last_l2y_pressed_ = l2y_pressed;

                bool imu_tilt = std::abs(low_state.imu_state().rpy()[0])>0.8 || std::abs(low_state.imu_state().rpy()[1])>0.8;
                bool damping_trigger = _firstLowCmdReceived && (l2b_edge);
                if(damping_trigger && !damping_mode_){
                    damping_mode_ = true;
                    damping_log_count_ = 0;
                    std::cout << "Switched to Damping Mode!" << std::endl;
                }

                if(damping_mode_){
                    for(int i=0; i<NUM_MOTOR; i++){
                        low_cmd.motor_cmd()[i].q() = 0;
                        low_cmd.motor_cmd()[i].dq() = 0;
                        low_cmd.motor_cmd()[i].kp() = 0;
                        low_cmd.motor_cmd()[i].kd() = 5;
                        low_cmd.motor_cmd()[i].tau() = 0;
                    }

                    // 同步对手部发送阻尼指令（kp=0，kd=0.2，关节置中）
                    dexWriter();

                    if(l2y_edge){
                        std::cout<< "L2+Y is pressed, recover to Policy Mode." <<std::endl;
                        time_ = 0.f;
                        damping_mode_ = false;
                    }else{
                        if(damping_log_count_ % 50 == 0){
                            std::cout<<"Press L2+Y to recover; L2+B to re-enter damping." <<std::endl;
                        }
                        damping_log_count_++;
                    }
                }else{
                    for(int i=0; i<NUM_MOTOR; i++){
                        low_cmd.motor_cmd()[i].q() = joint_command_simple.q_des[i];
                        low_cmd.motor_cmd()[i].dq() = joint_command_simple.qd_des[i];
                        low_cmd.motor_cmd()[i].kp() = joint_command_simple.kp[i];
                        low_cmd.motor_cmd()[i].kd() = joint_command_simple.kd[i];
                        low_cmd.motor_cmd()[i].tau() = joint_command_simple.tau_ff[i];
                    }
                }
            }
            
            low_cmd.crc() = Crc32Core((uint32_t *)&low_cmd, (sizeof(low_cmd)>>2)-1);
            lowcmd_publisher_->Write(low_cmd);
    }
};

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " networkInterface"<< std::endl;
    exit(-1);
  }

  std::cout << "Make sure the robot is hung up!" << std::endl
            << "You should not run the deploy code until the robot has moved to default positions!" <<std::endl
            << "You may press [L2 + B] to stop the process." <<std::endl
            << "You may press [L2 + Y] to recover from stopped state." << std::endl
            << "You may double press [L1 + B] for emergency termination." << std::endl
            << "Press Enter to continue ..." <<std::endl;
  std::cin.ignore(); // Press Enter to continue

  std::string networkInterface = argv[1];
  RobotController custom(networkInterface);
  while (true) usleep(20000); // 0.02s
  return 0;
}