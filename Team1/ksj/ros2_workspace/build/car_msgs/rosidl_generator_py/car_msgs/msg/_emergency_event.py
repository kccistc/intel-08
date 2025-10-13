# generated from rosidl_generator_py/resource/_idl.py.em
# with input from car_msgs:msg/EmergencyEvent.idl
# generated code does not contain a copyright notice

# This is being done at the module level and not on the instance level to avoid looking
# for the same variable multiple times on each instance. This variable is not supposed to
# change during runtime so it makes sense to only look for it once.
from os import getenv

ros_python_check_fields = getenv('ROS_PYTHON_CHECK_FIELDS', default='')


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_EmergencyEvent(type):
    """Metaclass of message 'EmergencyEvent'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'MSG_TYPE_EMERGENCY_BRAKE': 0,
        'MSG_TYPE_OBSTACLE_AHEAD': 1,
        'MSG_TYPE_RECKLESS_DRIVING': 2,
        'MSG_TYPE_UNKNOWN': 255,
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('car_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'car_msgs.msg.EmergencyEvent')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__emergency_event
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__emergency_event
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__emergency_event
            cls._TYPE_SUPPORT = module.type_support_msg__msg__emergency_event
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__emergency_event

            from geometry_msgs.msg import Point
            if Point.__class__._TYPE_SUPPORT is None:
                Point.__class__.__import_type_support__()

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'MSG_TYPE_EMERGENCY_BRAKE': cls.__constants['MSG_TYPE_EMERGENCY_BRAKE'],
            'MSG_TYPE_OBSTACLE_AHEAD': cls.__constants['MSG_TYPE_OBSTACLE_AHEAD'],
            'MSG_TYPE_RECKLESS_DRIVING': cls.__constants['MSG_TYPE_RECKLESS_DRIVING'],
            'MSG_TYPE_UNKNOWN': cls.__constants['MSG_TYPE_UNKNOWN'],
        }

    @property
    def MSG_TYPE_EMERGENCY_BRAKE(self):
        """Message constant 'MSG_TYPE_EMERGENCY_BRAKE'."""
        return Metaclass_EmergencyEvent.__constants['MSG_TYPE_EMERGENCY_BRAKE']

    @property
    def MSG_TYPE_OBSTACLE_AHEAD(self):
        """Message constant 'MSG_TYPE_OBSTACLE_AHEAD'."""
        return Metaclass_EmergencyEvent.__constants['MSG_TYPE_OBSTACLE_AHEAD']

    @property
    def MSG_TYPE_RECKLESS_DRIVING(self):
        """Message constant 'MSG_TYPE_RECKLESS_DRIVING'."""
        return Metaclass_EmergencyEvent.__constants['MSG_TYPE_RECKLESS_DRIVING']

    @property
    def MSG_TYPE_UNKNOWN(self):
        """Message constant 'MSG_TYPE_UNKNOWN'."""
        return Metaclass_EmergencyEvent.__constants['MSG_TYPE_UNKNOWN']


class EmergencyEvent(metaclass=Metaclass_EmergencyEvent):
    """
    Message class 'EmergencyEvent'.

    Constants:
      MSG_TYPE_EMERGENCY_BRAKE
      MSG_TYPE_OBSTACLE_AHEAD
      MSG_TYPE_RECKLESS_DRIVING
      MSG_TYPE_UNKNOWN
    """

    __slots__ = [
        '_header',
        '_msg_type',
        '_vehicle_id',
        '_position',
        '_confidence_score',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'msg_type': 'uint8',
        'vehicle_id': 'string',
        'position': 'geometry_msgs/Point',
        'confidence_score': 'float',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Point'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        if 'check_fields' in kwargs:
            self._check_fields = kwargs['check_fields']
        else:
            self._check_fields = ros_python_check_fields == '1'
        if self._check_fields:
            assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
                'Invalid arguments passed to constructor: %s' % \
                ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.msg_type = kwargs.get('msg_type', int())
        self.vehicle_id = kwargs.get('vehicle_id', str())
        from geometry_msgs.msg import Point
        self.position = kwargs.get('position', Point())
        self.confidence_score = kwargs.get('confidence_score', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.get_fields_and_field_types().keys(), self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    if self._check_fields:
                        assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.msg_type != other.msg_type:
            return False
        if self.vehicle_id != other.vehicle_id:
            return False
        if self.position != other.position:
            return False
        if self.confidence_score != other.confidence_score:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if self._check_fields:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def msg_type(self):
        """Message field 'msg_type'."""
        return self._msg_type

    @msg_type.setter
    def msg_type(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'msg_type' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'msg_type' field must be an unsigned integer in [0, 255]"
        self._msg_type = value

    @builtins.property
    def vehicle_id(self):
        """Message field 'vehicle_id'."""
        return self._vehicle_id

    @vehicle_id.setter
    def vehicle_id(self, value):
        if self._check_fields:
            assert \
                isinstance(value, str), \
                "The 'vehicle_id' field must be of type 'str'"
        self._vehicle_id = value

    @builtins.property
    def position(self):
        """Message field 'position'."""
        return self._position

    @position.setter
    def position(self, value):
        if self._check_fields:
            from geometry_msgs.msg import Point
            assert \
                isinstance(value, Point), \
                "The 'position' field must be a sub message of type 'Point'"
        self._position = value

    @builtins.property
    def confidence_score(self):
        """Message field 'confidence_score'."""
        return self._confidence_score

    @confidence_score.setter
    def confidence_score(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'confidence_score' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'confidence_score' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._confidence_score = value
