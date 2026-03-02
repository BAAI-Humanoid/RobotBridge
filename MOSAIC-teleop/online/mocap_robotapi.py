from ctypes import *
from collections import namedtuple
from platform import *
import time
import os

# 获取系统架构
architecture = machine()
# 定义so文件的存放路径
so_files_path = None
# 根据系统架构选择.so文件
if architecture == "arm64" or architecture == "aarch64":
    so_files_path = "./lib/librobotapi_arm64.so"
elif architecture == "x86_64":
    so_files_path = "./lib/librobotapi_x86-64.so"
elif architecture == "arm":
    so_files_path = "./lib/librobotapi_arm.so"    
else:
    print("Unsupported architecture:", architecture)
    so_files_path = None

MocapApi = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), {
  'DarWin': '',
  'Linux':  so_files_path,
  'Windows': './lib/MocapApi.dll'
}[system()]))

MCPError = namedtuple('EMCPError', [
  'NoError',
  'MoreEvent',
  'InsufficientBuffer',
  'InvalidObject',
  'InvalidHandle',
  'InvalidParameter',
  'NotSupported',
  'IgnoreUDPSettings',
  'IgnoreTCPSettings',
  'IgnoreBvhSettings',
  'JointNotFound',
  'WithoutTransformation',
  'NoneMessage',
  'NoneParent',
  'NoneChild',
  'AddressInUse'
])._make(range(16))

MCPJointTag = namedtuple('EMCPJointTag', [
  'Hips',
  'RightUpLeg',
  'RightLeg',
  'RightFoot',
  'LeftUpLeg',
  'LeftLeg',
  'LeftFoot',
  'Spine',
  'Spine1',
  'Spine2',
  'Neck',
  'Neck1',
  'Head',
  'RightShoulder',
  'RightArm',
  'RightForeArm',
  'RightHand',
  'RightHandThumb1',
  'RightHandThumb2',
  'RightHandThumb3',
  'RightInHandIndex',
  'RightHandIndex1',
  'RightHandIndex2',
  'RightHandIndex3',
  'RightInHandMiddle',
  'RightHandMiddle1',
  'RightHandMiddle2',
  'RightHandMiddle3',
  'RightInHandRing',
  'RightHandRing1',
  'RightHandRing2',
  'RightHandRing3',
  'RightInHandPinky',
  'RightHandPinky1',
  'RightHandPinky2',
  'RightHandPinky3',
  'LeftShoulder',
  'LeftArm',
  'LeftForeArm',
  'LeftHand',
  'LeftHandThumb1',
  'LeftHandThumb2',
  'LeftHandThumb3',
  'LeftInHandIndex',
  'LeftHandIndex1',
  'LeftHandIndex2',
  'LeftHandIndex3',
  'LeftInHandMiddle',
  'LeftHandMiddle1',
  'LeftHandMiddle2',
  'LeftHandMiddle3',
  'LeftInHandRing',
  'LeftHandRing1',
  'LeftHandRing2',
  'LeftHandRing3',
  'LeftInHandPinky',
  'LeftHandPinky1',
  'LeftHandPinky2',
  'LeftHandPinky3',
  'Spine3',
  'JointsCount',
])._make([x for x in range(61)])

MCPRigidBodyHandle = c_uint64


class MCPRigidBody(object):
  IMCPRigidBodyApi_Version = c_char_p(b'PROC_TABLE:IMCPRigidBody_001')

  class MCPRigidBodyApi(Structure):
    _fields_ = [
      ('GetRigidBodyRotation',
       CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPRigidBodyHandle)),
      ('GetRigidBodyPosition',
       CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPRigidBodyHandle)),
      ('GetRigidBodyStatus', CFUNCTYPE(c_int32, POINTER(c_int32), MCPRigidBodyHandle)),
      ('GetRigidBodyId', CFUNCTYPE(c_int32, POINTER(c_int32), MCPRigidBodyHandle))
    ]

  api = POINTER(MCPRigidBodyApi)()

  def __init__(self, rigid_body_handle):
    if not self.api:
      err = MocapApi.MCPGetGenericInterface(self.IMCPRigidBodyApi_Version, pointer(self.api))
      if err != MCPError.NoError:
        raise RuntimeError('Can not get MCPSensorModule interface: {0}'.format(MCPError._fields[err]))
    self.handle = rigid_body_handle

  def get_rotation(self):
    x = c_float()
    y = c_float()
    z = c_float()
    w = c_float()
    err = self.api.contents.GetRigidBodyRotation(pointer(x), pointer(y), pointer(z), pointer(w), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get rigid body rotation: {0}'.format(MCPError._fields[err]))
    return w.value, x.value, y.value, z.value

  def get_position(self):
    x = c_float()
    y = c_float()
    z = c_float()
    err = self.api.contents.GetRigidBodyPosition(pointer(x), pointer(y), pointer(z), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get rigid body position: {0}'.format(MCPError._fields[err]))
    return x.value, y.value, z.value

  def get_status(self):
    status = c_int32()
    err = self.api.contents.GetRigidBodyStatus(pointer(status), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get rigid body status: {0}'.format(MCPError._fields[err]))
    return status.value

  def get_id(self):
    rigid_id = c_int32()
    err = self.api.contents.GetRigidBodyId(pointer(rigid_id), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get rigid body id: {0}'.format(err))
    return rigid_id.value


MCPSensorModuleHandle = c_uint64


class MCPSensorModule(object):
  IMCPSensorModuleApi_Version = c_char_p(b'PROC_TABLE:IMCPSensorModule_001')

  class MCPSensorModuleApi(Structure):
    _fields_ = [
      ('GetSensorModulePosture',
       CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
                 MCPSensorModuleHandle)),
      ('GetSensorModuleAngularVelocity',
       CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPSensorModuleHandle)),
      ('GetSensorModuleAcceleratedVelocity',
       CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPSensorModuleHandle))
    ]

  api = POINTER(MCPSensorModuleApi)()

  def __init__(self, sensor_handle):
    if not self.api:
      err = MocapApi.MCPGetGenericInterface(self.IMCPSensorModuleApi_Version, pointer(self.api))
      if err != MCPError.NoError:
        raise RuntimeError('Can not get MCPSensorModule interface: {0}'.format(MCPError._fields[err]))
    self.handle = sensor_handle

  def get_posture(self):
    x = c_float()
    y = c_float()
    z = c_float()
    w = c_float()
    err = self.api.contents.GetSensorModulePosture(pointer(x), pointer(y), pointer(z), pointer(w), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get sensor module posture: {0}'.format(MCPError._fields[err]))
    return w.value, x.value, y.value, z.value

  def get_angular_velocity(self):
    x = c_float()
    y = c_float()
    z = c_float()
    err = self.api.contents.GetSensorModuleAngularVelocity(pointer(x), pointer(y), pointer(z), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get sensor module angular velocity: {0}'.format(MCPError._fields[err]))
    return x.value, y.value, z.value

  def get_accelerated_velocity(self):
    x = c_float()
    y = c_float()
    z = c_float()
    err = self.api.contents.GetSensorModuleAcceleratedVelocity(pointer(x), pointer(y), pointer(z), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get sensor module accelerated velocity: {0}'.format(MCPError._fields[err]))
    return x.value, y.value, z.value


MCPBodyPartHandle = c_uint64


class MCPBodyPart(object):
  IMCPBodyPartApi_Version = c_char_p(b'PROC_TABLE:IMCPBodyPart_001')

  class MCPBodyPartApi(Structure):
    _fields_ = [
      ('GetJointPosition', CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPBodyPartHandle)),
      ('GetJointDisplacementSpeed',
       CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPBodyPartHandle)),
      ('GetBodyPartPosture',
       CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPBodyPartHandle))
    ]

  api = POINTER(MCPBodyPartApi)()

  def __init__(self, body_part_handle):
    if not self.api:
      err = MocapApi.MCPGetGenericInterface(self.IMCPBodyPartApi_Version, pointer(self.api))
      if err != MCPError.NoError:
        raise RuntimeError('Can not get MCPBodyPartApi interface: {0}'.format(MCPError._fields[err]))
    self.handle = body_part_handle

  def get_position(self):
    x = c_float()
    y = c_float()
    z = c_float()
    err = self.api.contents.GetJointPosition(pointer(x), pointer(y), pointer(z), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint position:{0}'.format(MCPError._fileds[err]))
    return x.value, y.value, z.value

  def get_displacement_speed(self):
    x = c_float()
    y = c_float()
    z = c_float()
    err = self.api.contents.GetJointDisplacementSpeed(pointer(x), pointer(y), pointer(z), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint displacement speed: {0}'.format(MCPError._fields[err]))
    return x.value, y.value, z.value

  def get_posture(self):
    x = c_float()
    y = c_float()
    z = c_float()
    w = c_float()
    err = self.api.contents.GetBodyPartPosture(pointer(x), pointer(y), pointer(z), pointer(w), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get body part posture: {0}'.format(MCPError._fields[err]))
    return w.value, x.value, y.value, z.value


MCPJointHandle = c_uint64


class MCPJoint(object):
  IMCPJointApi_Version = c_char_p(b"PROC_TABLE:IMCPJoint_003")

  class MCPJointApi(Structure):
    _fields_ = [
      ('GetJointName', CFUNCTYPE(c_int32, POINTER(c_char_p), MCPJointHandle)),
      ('GetJointLocalRotation',
       CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPJointHandle)),
      ('GetJointLocalRotationByEuler',
       CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPJointHandle)),
      ('GetJointLocalTransformation',
       CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPJointHandle)),
      ('GetJointDefaultLocalTransformation',
       CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPJointHandle)),
      ('GetJointChild', CFUNCTYPE(c_int32, POINTER(MCPJointHandle), POINTER(c_uint32), MCPJointHandle)),
      ('GetJointBodyPart', CFUNCTYPE(c_int32, POINTER(MCPBodyPartHandle), MCPJointHandle)),
      ('GetJointSensorModule', CFUNCTYPE(c_int32, POINTER(MCPSensorModuleHandle), MCPJointHandle)),
      ('GetJointTag', CFUNCTYPE(c_int32, POINTER(c_int32), MCPJointHandle)),
      ('GetJointNameByTag', CFUNCTYPE(c_int32, POINTER(c_char_p), c_int32)),
      ('GetJointChildJointTag', CFUNCTYPE(c_int32, POINTER(c_int32), POINTER(c_uint32), c_int32)),
      ('GetJointParentJointTag', CFUNCTYPE(c_int32, POINTER(c_int32), c_int32))
    ]

  api = POINTER(MCPJointApi)()

  def __init__(self, joint_handle):
    if not self.api:
      err = MocapApi.MCPGetGenericInterface(self.IMCPJointApi_Version, pointer(self.api))
      if err != MCPError.NoError:
        raise RuntimeError('Can not get MCPJointApi interface: {0}'.format(MCPError._fields[err]))
    self.handle = joint_handle

  def get_name(self):
    joint_name = c_char_p()
    err = self.api.contents.GetJointName(pointer(joint_name), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint name: {0}'.format(MCPError._fields[err]))
    return str(joint_name.value, encoding='utf8')

  def get_local_rotation(self):
    x = c_float()
    y = c_float()
    z = c_float()
    w = c_float()
    err = self.api.contents.GetJointLocalRotation(pointer(x), pointer(y), pointer(z), pointer(w), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint local rotation: {0}'.format(MCPError._fields[err]))
    return w.value, x.value, y.value, z.value

  def get_local_rotation_by_euler(self):
    x = c_float()
    y = c_float()
    z = c_float()
    err = self.api.contents.GetJointLocalRotationByEuler(pointer(x), pointer(y), pointer(z), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint euler: {0}'.format(MCPError._fields[err]))
    return x.value, y.value, z.value

  def get_local_position(self):
    x = c_float()
    y = c_float()
    z = c_float()
    err = self.api.contents.GetJointLocalTransformation(pointer(x), pointer(y), pointer(z), self.handle)
    if err != MCPError.NoError:
      return None
    return x.value, y.value, z.value

  def get_default_local_position(self):
    x = c_float()
    y = c_float()
    z = c_float()
    err = self.api.contents.GetJointDefaultLocalTransformation(pointer(x), pointer(y), pointer(z), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint default local transformation: {0}'.format(MCPError._fields[err]))
    return x.value, y.value, z.value

  def get_children(self):
    joint_count = c_uint32()
    err = self.api.contents.GetJointChild(POINTER(MCPJointHandle)(), pointer(joint_count), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint child count: {0}'.format(MCPError._fields[err]))
    joint_handles = (MCPJointHandle * joint_count.value)()
    err = self.api.contents.GetJointChild(joint_handles, pointer(joint_count), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint child: {0}'.format(MCPError._fields[err]))
    return [MCPJoint(joint_handles[i]) for i in range(joint_count.value)]

  def get_body_part(self):
    body_part_handle = MCPBodyPartHandle()
    err = self.api.contents.GetJointBodyPart(pointer(body_part_handle), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint body part: {0}'.format(MCPError._fields[err]))
    return MCPBodyPart(body_part_handle)

  def get_sensor_module(self):
    sensor_handle = MCPSensorModuleHandle()
    err = self.api.contents.GetJointSensorModule(pointer(sensor_handle), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get sensor module: {0}'.format(err))
    return MCPSensorModule(sensor_handle)

  def get_tag(self):
    tag = c_int32()
    err = self.api.contents.GetJointTag(pointer(tag), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint tag: {0}'.format(MCPError._fields[err]))
    return tag.value

  def get_name_by_tag(self, tag):
    joint_name = c_char_p()
    err = self.api.contents.GetJointNameByTag(pointer(joint_name), c_int32(tag))
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint name of joint tag {0}: {1}'.format(tag, err))
    return str(joint_name.value, encoding='utf8')

  def get_child_joint_tag(self, tag):
    joint_count = c_uint32()
    err = self.api.contents.GetJointChildJointTag(POINTER(c_int32)(), pointer(joint_count), c_int32(tag))
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint child joint tag: {0}'.format(MCPError._fields[err]))
    joints_tag = (c_int32 * joint_count.value)()
    err = self.api.contents.GetJointChildJointTag(joints_tag, pointer(joint_count), c_int32(tag))
    if err != MCPError.NoError:
      raise RuntimeError('Can not get joint child joint tag: {0}'.format(MCPError._fields[err]))
    return [joints_tag[i].value for i in range(joint_count.value)]

  def get_parent_joint_tag(self, tag):
    joint_tag = c_int32()
    err = self.api.contents.GetJointParentJointTag(pointer(joint_tag), c_int32(tag))
    if err != MCPError.NoError:
      raise RuntimeError('can not get joint parent tag: {0}'.format(MCPError._fields[err]))
    return joint_tag.value


MCPAvatarHandle = c_uint64


class MCPAvatar(object):
  IMCPAvatarApi_Version = c_char_p(b'PROC_TABLE:IMCPAvatar_003')

  class MCPAvatarApi(Structure):
    _fields_ = [
      ('GetAvatarIndex', CFUNCTYPE(c_int32, POINTER(c_uint32), MCPAvatarHandle)),
      ('GetAvatarRootJoint', CFUNCTYPE(c_int32, POINTER(MCPJointHandle), MCPAvatarHandle)),
      ('GetAvatarJoints', CFUNCTYPE(c_int32, POINTER(MCPJointHandle), POINTER(c_uint32), MCPAvatarHandle)),
      ('GetAvatarJointByName', CFUNCTYPE(c_int32, c_char_p, POINTER(MCPJointHandle), MCPAvatarHandle)),
      ('GetAvatarName', CFUNCTYPE(c_int32, POINTER(c_char_p), MCPAvatarHandle)),
      ('GetAvatarRigidBodies', CFUNCTYPE(c_int32, POINTER(MCPRigidBodyHandle), POINTER(c_uint32), MCPAvatarHandle)),
      ('GetAvatarJointHierarchy', CFUNCTYPE(c_int32, POINTER(c_char_p))),
      ('GetAvatarPostureIndex', CFUNCTYPE(c_int32, POINTER(c_uint32), POINTER(MCPAvatarHandle))),
      ('GetAvatarPostureTimeCode',
       CFUNCTYPE(c_int32, POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32),
                 POINTER(MCPAvatarHandle))),
    ]

  api = POINTER(MCPAvatarApi)()

  def __init__(self, avatar_handle):
    if not self.api:
      err = MocapApi.MCPGetGenericInterface(self.IMCPAvatarApi_Version, pointer(self.api))
      if err != MCPError.NoError:
        raise RuntimeError('Can not get MCPAvatar interface: {0}'.format(MCPError._fields[err]))
    self.handle = avatar_handle

  def get_index(self):
    index = c_uint32(0)
    err = self.api.contents.GetAvatarIndex(pointer(index), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get avatar index: {0}'.format(MCPError._fields[err]))
    return index.value

  def get_root_joint(self):
    joint_handle = MCPJointHandle()
    err = self.api.contents.GetAvatarRootJoint(pointer(joint_handle), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get avatar root joint: {0}'.format(err))
    return MCPJoint(joint_handle)

  def get_joints(self):
    joint_count = c_uint32()
    err = self.api.contents.GetAvatarJoints(POINTER(MCPJointHandle)(), pointer(joint_count), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get avatar joints: {0}'.format(MCPError._fields[err]))
    joints_handle = (MCPJointHandle * joint_count.value)()
    err = self.api.contents.GetAvatarJoints(joints_handle, pointer(joint_count), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get avatar Joints: {0}'.format(MCPError._fields[err]))
    return [MCPJoint(joints_handle[i]) for i in range(joint_count.value)]

  def get_joint_by_name(self, name):
    joint_name = c_char_p(bytes(name, encoding='utf8'))
    joint_handle = MCPJointHandle()
    err = self.api.contents.GetAvatarJointByName(joint_name, pointer(joint_handle), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get avatar Joints: {0}'.format(MCPError._fields[err]))
    return MCPJoint(joint_handle)

  def get_name(self):
    avatar_name = c_char_p()
    err = self.api.contents.GetAvatarName(pointer(avatar_name), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get avatar name: {0}'.format(MCPError._fields[err]))
    return str(avatar_name.value, encoding='utf8')

  def get_rigid_bodies(self):
    rigid_body_count = c_uint32()
    err = self.api.contents.GetAvatarRigidBodies(POINTER(MCPRigidBodyHandle)(), pointer(rigid_body_count), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get avatar rigid bodies: {0}'.format(MCPError._fields[err]))
    rigid_body_handles = (MCPRigidBodyHandle * rigid_body_count.value)()
    err = self.api.contents.GetAvatarRigidBodies(rigid_body_handles, pointer(rigid_body_count), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get avatar rigid bodies: {0}'.format(MCPError._fields[err]))
    return [MCPRigidBody(rigid_body_handles[i]) for i in range(rigid_body_count.value)]

  def get_joint_hierarchy(self):
    hierarchy = c_char_p()
    err = self.api.contents.GetAvatarJointHierarchy(pointer(hierarchy))
    if err != MCPError.NoError:
      raise RuntimeError('Can not get avatar joint hierarchy: {0}'.format(err))
    return str(hierarchy.value, encoding='utf8')


class MCPEventDataReserved(Structure):
  _fields_ = [
    ('reserved0', c_uint64),
    ('reserved1', c_uint64),
    ('reserved2', c_uint64),
    ('reserved3', c_uint64),
    ('reserved4', c_uint64),
    ('reserved5', c_uint64),
  ]


class MCPEventData(Union):
  _fields_ = [
    ('reserved', MCPEventDataReserved),
    ('avatar_handle', MCPAvatarHandle),
    ('error', c_int32)
  ]


MCPEventType = namedtuple('EMCPEventType', [
  'InvalidEvent',
  'AvatarUpdated',
  'RigidBodyUpdated',
  'Error'
])(0, 256, 512, 768)


class MCPEvent(Structure):
  _fields_ = [
    ("size", c_uint32),
    ("event_type", c_int32),
    ('timestamp', c_double),
    ("event_data", MCPEventData)
  ]


MCPBvhRotation = namedtuple('EMCPBvhRotation', [
  'XYZ',
  'XZY',
  'YXZ',
  'YZX',
  'ZXY',
  'ZYX'
])(0, 1, 2, 3, 4, 5)

MCPBvhData = namedtuple('EMCPBvhData', [
  'String',
  'BinaryWithOldFrameHeader',
  'Binary',
  'BinaryLegacyHumanHierarchy'
])(0, 1, 2, 6)

MCPBvhDisplacement = namedtuple('EMCPBvhDisplacement', [
  'Disable',
  'Enable'
])(0, 1)

MCPSettingsHandle = c_uint64


class MCPSettings(object):
  IMCPSettingsApi_Version = c_char_p(b'PROC_TABLE:IMCPSettings_001')

  class MCPSettingsApi(Structure):
    _fields_ = [
      ('CreateSettings', CFUNCTYPE(c_int32, POINTER(MCPSettingsHandle))),
      ('DestroySettings', CFUNCTYPE(c_int32, MCPSettingsHandle)),
      ('SetSettingsUDP', CFUNCTYPE(c_int32, c_uint16, MCPSettingsHandle)),
      ('SetSettingsTCP', CFUNCTYPE(c_int32, c_char_p, c_uint16, MCPSettingsHandle)),
      ('SetSettingsBvhRotation', CFUNCTYPE(c_int32, c_int32, MCPSettingsHandle)),
      ('SetSettingsBvhDisplacement', CFUNCTYPE(c_int32, c_int32, MCPSettingsHandle)),
      ('SetSettingsBvhData', CFUNCTYPE(c_int32, c_int32, MCPSettingsHandle)),
      ('SetSettingsCalcData', CFUNCTYPE(c_int32, MCPSettingsHandle))
    ]

  api = POINTER(MCPSettingsApi)()

  def __init__(self):
    if not self.api:
      err = MocapApi.MCPGetGenericInterface(self.IMCPSettingsApi_Version, pointer(self.api))
      if err != MCPError.NoError:
        raise RuntimeError('Can not get MCPSettings interface: {0}'.format(MCPError._fields[err]))
    self.handle = MCPSettingsHandle()
    err = self.api.contents.CreateSettings(pointer(self.handle))
    if err != MCPError.NoError:
      raise RuntimeError('Can not create settings: {0}'.format(MCPError._fields[err]))

  def __del__(self):
    err = self.api.contents.DestroySettings(self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not destroy settings: {0}'.format(MCPError._fields[err]))

  def set_udp(self, local_port):
    err = self.api.contents.SetSettingsUDP(c_uint16(local_port), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set udp port of {0}: {1}'.format(local_port, MCPError._fields[err]))

  def set_tcp(self, ip, port):
    err = self.api.contents.SetSettingsTCP(c_char_p(bytes(ip, encoding='utf8')), c_uint16(port), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not settings tcp addr of {0}:{1}: {2}'.format(ip, port, MCPError._fields[err]))

  def set_bvh_rotation(self, bvh_rotation):
    err = self.api.contents.SetSettingsBvhRotation(c_int32(bvh_rotation), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set bvh rotation: {0}'.format(MCPError._fields[err]))

  def set_bvh_displacement(self, bvh_transformation):
    err = self.api.contents.SetSettingsBvhDisplacement(c_int32(bvh_transformation), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set bvh displacement: {0}'.format(MCPError._fields[err]))

  def set_bvh_data(self, bvh_data):
    err = self.api.contents.SetSettingsBvhData(c_int32(bvh_data), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set bvh data: {0}'.format(MCPError._fields[err]))

  def set_calc_data(self):
    err = self.api.contents.SetSettingsCalcData(self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set calc data: {0}'.format(MCPError._fields[err]))


MCPUpVector = namedtuple('EMCPUpVector', [
  'XAxis',
  'YAxis',
  'ZAxis'
])(1, 2, 3)

MCPFrontVector = namedtuple('EMCPFrontVector', [
  'ParityEven',
  'ParityOdd'
])(1, 2)

MCPCoordSystem = namedtuple('EMCPCoordSystem', [
  'RightHanded',
  'LeftHanded'
])(0, 1)

MCPRotatingDirection = namedtuple('EMCPRotatingDirection', [
  'Clockwise',
  'CounterClockwise'
])(0, 1)

MCPPreDefinedRenderSettings = namedtuple('EMCPPreDefinedRenderSettings', [
  'Default',
  'UnrealEngine',
  'Unity3D',
  'Count'
])(0, 1, 2, 3)

MCPUnit = namedtuple('EMCPUnit', [
  'Centimeter',
  'Meter'
])(0, 1)

MCPRenderSettingsHandle = c_uint64


class MCPRenderSettings(object):
  IMCPRenderSettingsApi_Version = c_char_p(b'PROC_TABLE:IMCPRenderSettings_001')

  class MCPRenderSettingsApi(Structure):
    _fields_ = [
      ('CreateRenderSettings', CFUNCTYPE(c_int32, POINTER(MCPRenderSettingsHandle))),
      ('GetPreDefRenderSettings', CFUNCTYPE(c_int32, c_int32, POINTER(MCPRenderSettingsHandle))),
      ('SetUpVector', CFUNCTYPE(c_int32, c_int32, c_int32, MCPRenderSettingsHandle)),
      ('GetUpVector', CFUNCTYPE(c_int32, POINTER(c_int32), POINTER(c_int32), MCPRenderSettingsHandle)),
      ('SetFrontVector', CFUNCTYPE(c_int32, c_int32, c_int32, MCPRenderSettingsHandle)),
      ('GetFrontVector', CFUNCTYPE(c_int32, POINTER(c_int32), POINTER(c_int32), MCPRenderSettingsHandle)),
      ('SetCoordSystem', CFUNCTYPE(c_int32, c_int32, MCPRenderSettingsHandle)),
      ('GetCoordSystem', CFUNCTYPE(c_int32, POINTER(c_int32), MCPRenderSettingsHandle)),
      ('SetRotatingDirection', CFUNCTYPE(c_int32, c_int32, MCPRenderSettingsHandle)),
      ('GetRotationDirection', CFUNCTYPE(c_int32, POINTER(c_int32), MCPRenderSettingsHandle)),
      ('SetUnit', CFUNCTYPE(c_int32, c_int32, MCPRenderSettingsHandle)),
      ('GetUnit', CFUNCTYPE(c_int32, POINTER(c_int32), MCPRenderSettingsHandle)),
      ('DestroyRenderSettings', CFUNCTYPE(c_int32, MCPRenderSettingsHandle))
    ]

  api = POINTER(MCPRenderSettingsApi)()

  def __init__(self, pre_def=None):
    if not self.api:
      err = MocapApi.MCPGetGenericInterface(self.IMCPRenderSettingsApi_Version, pointer(self.api))
      if err != MCPError.NoError:
        raise RuntimeError('Can not get MCPRenderSettings interface: {0}'.format(MCPError._fields[err]))
    self.pre_def = pre_def
    self.handle = MCPRenderSettingsHandle()
    if self.pre_def == None:
      err = self.api.contents.CreateRenderSettings(pointer(self.handle))
      if err != MCPError.NoError:
        raise RuntimeError('Can not create render settings: {0}'.format(MCPError._fields[err]))
    else:
      err = self.api.contents.GetPreDefRenderSettings(c_int32(pre_def), pointer(self.handle))
      if err != MCPError.NoError:
        raise RuntimeError('Can not get render settings: {0}'.format(MCPError._fields[err]))

  def __del__(self):
    if self.pre_def == None:
      err = self.api.contents.DestroyRenderSettings(self.handle)
      if err != MCPError.NoError:
        raise RuntimeError('Can not destroy render settings: {0}'.format(MCPError._fields[err]))

  def set_up_vector(self, up_vector, sign):
    err = self.api.contents.SetUpVector(c_int32(up_vector), c_int32(sign), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set up vector: {0}'.format(MCPError._fields[err]))

  def get_up_vector(self):
    up_vector = c_int32()
    sign = c_int32()
    err = self.api.contents.GetUpVector(pointer(up_vector), pointer(sign), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get up vector: {0}'.format(MCPError._fields[err]))
    return up_vector.value, sign.value

  def set_front_vector(self, front_vector, sign):
    err = self.api.contents.SetFrontVector(c_int32(front_vector), c_int32(sign), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set front vector: {0}'.format(MCPError._fields[err]))

  def get_front_vector(self):
    front_vector = c_int32()
    sign = c_int32()
    err = self.api.contents.GetFrontVector(pointer(front_vector), pointer(sign), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get front vector: {0}'.format(MCPError._fields[err]))
    return front_vector.value, sign.value

  def set_coord_system(self, coord_sys):
    err = self.api.contents.SetCoordSystem(c_int32(coord_sys), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set coord system: {0}'.format(MCPError._fields[err]))

  def get_coord_system(self):
    coord_sys = c_int32()
    err = self.api.contents.GetCoordSystem(pointer(coord_sys), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get coord system: {0}'.format(MCPError._fields[err]))
    return coord_sys.value

  def set_rotating_direction(self, rotating_direction):
    err = self.api.contents.SetRotatingDirection(c_int32(rotating_direction), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set rotating direction: {0}'.format(MCPError._fields[err]))

  def get_rotating_direction(self):
    rotating_direction = c_int32()
    err = self.api.contents.GetRotatingDirection(pointer(rotating_direction), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get rotating direction: {0}'.format(MCPError._fields[err]))
    return rotating_direction.value

  def set_unit(self, unit):
    err = self.api.contents.SetUnit(c_int32(unit), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set unit: {0}'.format(MCPError._fields[err]))

  def get_unit(self):
    unit = c_int32()
    err = self.api.contents.GetUnit(pointer(unit), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get unit: {0}'.format(MCPError._fields[err]))
    return unit.value

MCPRobotJointType=namedtuple('EMCPRobotJointType',
    ['JointLeftShoulder_Pitch',
        'JointLeftShoulder_Roll',
        'JointLeftShoulder_Yaw',
        'JointLeftElbow',
        'JointLeftWrist_Yaw',
        'JointLeftWrist_Roll',
        'JointLeftWrist_Pitch',
        'JointRightShoulder_Pitch',
        'JointRightShoulder_Roll',
        'JointRightShoulder_Yaw',
        'JointRightElbow',
        'JointRightWrist_Yaw',
        'JointRightWrist_Roll',
        'JointRightWrist_Pitch',
        'JointLeftHip_Yaw',
        'JointLeftHip_Pitch',
        'JointLeftHip_Roll',
        'JointLeftKnee',
        'JointLeftAnkle_Pitch',
        'JointLeftAnkle_Roll',
        'JointRightHip_Yaw',
        'JointRightHip_Pitch',
        'JointRightHip_Roll',
        'JointRightKnee',
        'JointRightAnkle_Pitch',
        'JointRightAnkle_Roll',
        'JointSpine_Yaw',
        'JointSpine_Roll',
        'JointSpine_Pitch',
        'JointHead_Yaw',
        'JointHead_Roll',
        'JointHead_Pitch',
        'JointCount'])._make([x for x in range(0,33)])

MCPRobotHandle = c_uint64

class MCPRobot(object):
  IMCPRobot_Version = c_char_p(b"PROC_TABLE:IMCPRobot_001")
  class MCPRobotApi(Structure):
    _fields_=[
      ('CreateRobot', CFUNCTYPE(c_int32,c_char_p,POINTER(MCPRobotHandle))),
      ('SetRobotFPS', CFUNCTYPE(c_int32, c_int32, MCPRobotHandle)),
      ('UpdateRobot', CFUNCTYPE(c_int32, MCPAvatarHandle, MCPRobotHandle)),
      ('GetRobotRawJointAngle', CFUNCTYPE(c_int32, c_int32, POINTER(c_float), MCPRobotHandle)),
      ('GetRobotRetargetJointAngle', CFUNCTYPE(c_int32, c_char_p, POINTER(c_float), MCPRobotHandle)),
      ('GetRobotRosFrameJson', CFUNCTYPE(c_int32, POINTER(c_char_p), c_bool, MCPRobotHandle)),
      ('GetRobotRootRotation', CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPRobotHandle)),
      ('GetRobotRootPosition', CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPRobotHandle)),
      ('DestroyRobot', CFUNCTYPE(c_int32, MCPRobotHandle)),
      ('RunRobotStep', CFUNCTYPE(c_int32, MCPRobotHandle))
       ]
  api = POINTER(MCPRobotApi)()
  def __init__(self, json_robot_desc):
    if not self.api:
      err = MocapApi.MCPGetGenericInterface(self.IMCPRobot_Version, pointer(self.api))
      if err != MCPError.NoError:
        raise  RuntimeError('Can not get MCPRobot interface: {}'.format(MCPError._fields[err]))
    self.handle = MCPSettingsHandle()
    err = self.api.contents.CreateRobot(c_char_p(bytes(json_robot_desc, encoding='utf8')), pointer(self.handle))
    if err != MCPError.NoError:
      raise  RuntimeError('Can not create robot: {0}'.format(MCPError._fields[err]))
  def __del__(self):
    err = self.api.contents.DestroyRobot(self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not destroy robot : {0}'.format(MCPError._fields[err]))

  def set_robot_fps(self,fps):
    err = self.api.contents.SetRobotFPS(c_uint32(fps),self.handle)
    if self != MCPError.NoError:
      raise RuntimeError('Can not set robot fps: {0}'.format(MCPError._fields[err]))
  def update_robot(self,avatar):
    err = self.api.contents.UpdateRobot(avatar.handle, self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not update robot: {0}'.format(MCPError._fields[err]))
  def get_robot_raw_joint_angle(self,robotJointType):
    val = c_float()
    err = self.api.contents.GetRobotRawJointAngle(c_int32(robotJointType),pointer(val), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get robot raw joint angle: {0}'.format(MCPError._fields[err]))
    return val.value
  def get_robot_retarget_joint_angle(self,jointName):
    val = c_float()
    err = self.api.contents.GetRobotRetargetJointAngle(c_char_p(bytes(jointName,encoding='utf8')), pointer(val),self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get robot retarget joint angle: {0}'.format(MCPError._fields[err]))
    return val.value
  def get_robot_ros_frame_json(self):
    val = c_float()
    json_str = c_char_p()
    err = self.api.contents.GetRobotRosFrameJson(pointer(json_str), pointer(val), self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get robot ros frame json: {0}'.format(MCPError._fields[err]))
    return str(json_str.value, encoding='utf8'),val.value
  def get_robot_root_rotataion(self):
    x = c_float()
    y = c_float()
    z = c_float()
    w = c_float()
    err = self.api.contents.GetRobotRootRotation(pointer(x),pointer(y),pointer(z),pointer(w),self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get robot robot rotation:{}'.format(MCPError._fields[err]))
    return x.value, y.value, z.value, w.value
  def get_robot_root_position(self):
    x = c_float()
    y = c_float()
    z = c_float()
    err = self.api.contents.GetRobotRootPosition(pointer(x), pointer(y), pointer(z),self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get robot root position:{}'.format(MCPError._fields[err]))
    return x.value, y.value, z.value
  def run_robot_step(self):
    err = self.api.contents.RunRobotStep(self.handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not run robot step:{}'.format(MCPError._fields[err]))


MCPApplicationHandle = c_uint64

class MCPApplication(object):
  IMCPApplicationApi_Version = c_char_p(b'PROC_TABLE:IMCPApplication_002')

  class MCPApplicationApi(Structure):
    _fields_ = [
      ('CreateApplication', CFUNCTYPE(c_int32, POINTER(MCPApplicationHandle))),
      ('DestroyApplication', CFUNCTYPE(c_int32, MCPApplicationHandle)),
      ('SetApplicationSettings', CFUNCTYPE(c_int32, MCPSettingsHandle, MCPApplicationHandle)),
      ('SetApplicationRenderSettings', CFUNCTYPE(c_int32, MCPRenderSettingsHandle, MCPApplicationHandle)),
      ('OpenApplication', CFUNCTYPE(c_int32, MCPApplicationHandle)),
      ('EnableApplicationCacheEvents', CFUNCTYPE(c_int32, MCPApplicationHandle)),
      ('DisableApplicationCacheEvents', CFUNCTYPE(c_int32, MCPApplicationHandle)),
      ('ApplicationCacheEventsIsEnabled', CFUNCTYPE(c_int32, POINTER(c_bool), MCPApplicationHandle)),
      ('CloseApplication', CFUNCTYPE(c_int32, MCPApplicationHandle)),
      ('GetApplicationRigidBodies', CFUNCTYPE(c_int32, POINTER(c_uint64), POINTER(c_uint32), MCPApplicationHandle)),
      ('GetApplicationAvatars', CFUNCTYPE(c_int32, POINTER(c_uint64), POINTER(c_uint32), MCPApplicationHandle)),
      ('PollApplicationNextEvent', CFUNCTYPE(c_int32, POINTER(MCPEvent), POINTER(c_uint32), MCPApplicationHandle))
    ]

  api = POINTER(MCPApplicationApi)()

  def __init__(self):
    if not self.api:
      err = MocapApi.MCPGetGenericInterface(self.IMCPApplicationApi_Version, pointer(self.api))
      if err != MCPError.NoError:
        raise RuntimeError('Can not get IMCPApplication interface: {0}'.format(MCPError._fields[err]))
    self._handle = MCPApplicationHandle()
    err = self.api.contents.CreateApplication(pointer(self._handle))
    if err != MCPError.NoError:
      raise RuntimeError('Can not create application: {0}'.format(MCPError._fields[err]))
    self._is_opened = False

  def __del__(self):
    err = self.api.contents.DestroyApplication(self._handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not destroy application: {0}'.format(MCPError._fields[err]))

  def set_settings(self, settings):
    err = self.api.contents.SetApplicationSettings(settings.handle, self._handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set application settings: {0}'.format(MCPError._fields[err]))

  def set_render_settings(self, settings):
    err = self.api.contents.SetApplicationRenderSettings(settings.handle, self._handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not set application render settings: {0}'.format(MCPError._fields[err]))

  def open(self):
    err = self.api.contents.OpenApplication(self._handle)
    self._is_opened = (err == MCPError.NoError)
    return err == MCPError.NoError, MCPError._fields[err]

  def is_opened(self):
    return self._is_opened

  def enable_event_cache(self):
    err = self.api.contents.EnableApplicationCacheEvents(self._handle)
    return err == MCPError.NoError, MCPError._fields[err]

  def disable_event_cache(self):
    err = self.api.contents.DisableApplicationCacheEvents(self._handle)
    return err == MCPError.NoError, MCPError._fields[err]

  def is_event_cache_enabled(self):
    enable = c_bool()
    err = self.api.contents.ApplicationCacheEventsIsEnabled(pointer(enable), self._handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get application event cache settings: {0}'.format(MCPError._fields[err]))
    return enable.value

  def close(self):
    err = self.api.contents.CloseApplication(self._handle)
    self._is_opened = False
    return err == MCPError.NoError, MCPError._fields[err]

  def get_rigid_bodies(self):
    rigid_body_size = c_uint32()
    err = self.api.contents.GetApplicationRigidBodies(POINTER(MCPRigidBodyHandle)(), pointer(rigid_body_size),
                                                      self._handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get application rigid bodies: {0}'.format(MCPError._fields[err]))
    rigid_body_handles = (MCPRigidBodyHandle * rigid_body_size.value)()
    err = self.api.GetApplicationRigidBodies(rigid_body_handles, pointer(rigid_body_size), self._handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get application rigid bodies: {0}'.format(MCPError._fields[err]))
    return [MCPRigidBody(rigid_body_handles[i]) for i in range(rigid_body_size.value)]

  def get_avatars(self):
    avatar_count = c_uint32()
    err = self.api.contents.GetApplicationAvatars(POINTER(MCPAvatarHandle)(), pointer(avatar_count), self._handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get application avatars: {0}'.format(MCPError._fields[err]))
    avatar_handles = (MCPAvatarHandle * avatar_count.value)()
    err = self.api.contents.GetApplicationAvatars(avatar_handles, pointer(avatar_count), self._handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not get application avatars: {0}'.format(MCPError._fields[err]))
    return [MCPAvatar(avatar_handles[i]) for i in range(avatar_count.value)]

  def poll_next_event(self):
    evt_count = c_uint32(0)
    err = self.api.contents.PollApplicationNextEvent(POINTER(MCPEvent)(), pointer(evt_count), self._handle)
    if err != MCPError.NoError:
      raise RuntimeError('Can not poll application next event: {0}'.format(MCPError._fields[err]))
    if evt_count.value == 0:
      return []
    evt_array = (MCPEvent * evt_count.value)()
    for i in range(evt_count.value):
      evt_array[i].size = sizeof(MCPEvent)
    err = self.api.contents.PollApplicationNextEvent(evt_array, pointer(evt_count), self._handle)
    if err != MCPError.NoError:
      return []
    if evt_count.value == 0:
      return []
    return [evt_array[i] for i in range(evt_count.value)]

def test_mocap_api():
  def print_joint(joint):
    print(joint.get_name())
    print(joint.get_local_rotation())
    print(joint.get_local_rotation_by_euler())
    print(joint.get_local_position())

    children = joint.get_children()
    for child in children:
      print_joint(child)

  robot = MCPRobot(open('retarget.json').read())
  app = MCPApplication()
  settings = MCPSettings()
  settings.set_udp(7012)
  app.set_settings(settings)
  app.open()
  while True:
    evts = app.poll_next_event()
    for evt in evts:
      if evt.event_type == MCPEventType.AvatarUpdated:
        avatar = MCPAvatar(evt.event_data.avatar_handle)
        robot.update_robot(avatar)
        robot.run_robot_step()
        print (robot.get_robot_ros_frame_json())
        # print(avatar.get_index())
        # print(avatar.get_name())
        # print_joint(avatar.get_root_joint())
      elif evt.event_type == MCPEventType.RigidBodyUpdated:
        print('rigid body updated')
      else:
        print('unknow event')

    time.sleep(0.001)

if __name__ == '__main__':
  test_mocap_api()