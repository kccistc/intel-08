# generated from rosidl_generator_py/resource/_idl.py.em
# with input from car_msgs:msg/V2VAlert.idl
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


class Metaclass_V2VAlert(type):
    """Metaclass of message 'V2VAlert'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
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
                'car_msgs.msg.V2VAlert')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__v2_v_alert
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__v2_v_alert
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__v2_v_alert
            cls._TYPE_SUPPORT = module.type_support_msg__msg__v2_v_alert
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__v2_v_alert

            from builtin_interfaces.msg import Time
            if Time.__class__._TYPE_SUPPORT is None:
                Time.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class V2VAlert(metaclass=Metaclass_V2VAlert):
    """Message class 'V2VAlert'."""

    __slots__ = [
        '_ver',
        '_src',
        '_seq',
        '_ts',
        '_type',
        '_severity',
        '_distance_m',
        '_road',
        '_lat',
        '_lon',
        '_suggest',
        '_ttl_s',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'ver': 'uint32',
        'src': 'string',
        'seq': 'uint32',
        'ts': 'builtin_interfaces/Time',
        'type': 'string',
        'severity': 'string',
        'distance_m': 'float',
        'road': 'string',
        'lat': 'double',
        'lon': 'double',
        'suggest': 'string',
        'ttl_s': 'float',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
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
        self.ver = kwargs.get('ver', int())
        self.src = kwargs.get('src', str())
        self.seq = kwargs.get('seq', int())
        from builtin_interfaces.msg import Time
        self.ts = kwargs.get('ts', Time())
        self.type = kwargs.get('type', str())
        self.severity = kwargs.get('severity', str())
        self.distance_m = kwargs.get('distance_m', float())
        self.road = kwargs.get('road', str())
        self.lat = kwargs.get('lat', float())
        self.lon = kwargs.get('lon', float())
        self.suggest = kwargs.get('suggest', str())
        self.ttl_s = kwargs.get('ttl_s', float())

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
        if self.ver != other.ver:
            return False
        if self.src != other.src:
            return False
        if self.seq != other.seq:
            return False
        if self.ts != other.ts:
            return False
        if self.type != other.type:
            return False
        if self.severity != other.severity:
            return False
        if self.distance_m != other.distance_m:
            return False
        if self.road != other.road:
            return False
        if self.lat != other.lat:
            return False
        if self.lon != other.lon:
            return False
        if self.suggest != other.suggest:
            return False
        if self.ttl_s != other.ttl_s:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def ver(self):
        """Message field 'ver'."""
        return self._ver

    @ver.setter
    def ver(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'ver' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'ver' field must be an unsigned integer in [0, 4294967295]"
        self._ver = value

    @builtins.property
    def src(self):
        """Message field 'src'."""
        return self._src

    @src.setter
    def src(self, value):
        if self._check_fields:
            assert \
                isinstance(value, str), \
                "The 'src' field must be of type 'str'"
        self._src = value

    @builtins.property
    def seq(self):
        """Message field 'seq'."""
        return self._seq

    @seq.setter
    def seq(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'seq' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'seq' field must be an unsigned integer in [0, 4294967295]"
        self._seq = value

    @builtins.property
    def ts(self):
        """Message field 'ts'."""
        return self._ts

    @ts.setter
    def ts(self, value):
        if self._check_fields:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'ts' field must be a sub message of type 'Time'"
        self._ts = value

    @builtins.property  # noqa: A003
    def type(self):  # noqa: A003
        """Message field 'type'."""
        return self._type

    @type.setter  # noqa: A003
    def type(self, value):  # noqa: A003
        if self._check_fields:
            assert \
                isinstance(value, str), \
                "The 'type' field must be of type 'str'"
        self._type = value

    @builtins.property
    def severity(self):
        """Message field 'severity'."""
        return self._severity

    @severity.setter
    def severity(self, value):
        if self._check_fields:
            assert \
                isinstance(value, str), \
                "The 'severity' field must be of type 'str'"
        self._severity = value

    @builtins.property
    def distance_m(self):
        """Message field 'distance_m'."""
        return self._distance_m

    @distance_m.setter
    def distance_m(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'distance_m' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'distance_m' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._distance_m = value

    @builtins.property
    def road(self):
        """Message field 'road'."""
        return self._road

    @road.setter
    def road(self, value):
        if self._check_fields:
            assert \
                isinstance(value, str), \
                "The 'road' field must be of type 'str'"
        self._road = value

    @builtins.property
    def lat(self):
        """Message field 'lat'."""
        return self._lat

    @lat.setter
    def lat(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'lat' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'lat' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._lat = value

    @builtins.property
    def lon(self):
        """Message field 'lon'."""
        return self._lon

    @lon.setter
    def lon(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'lon' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'lon' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._lon = value

    @builtins.property
    def suggest(self):
        """Message field 'suggest'."""
        return self._suggest

    @suggest.setter
    def suggest(self, value):
        if self._check_fields:
            assert \
                isinstance(value, str), \
                "The 'suggest' field must be of type 'str'"
        self._suggest = value

    @builtins.property
    def ttl_s(self):
        """Message field 'ttl_s'."""
        return self._ttl_s

    @ttl_s.setter
    def ttl_s(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'ttl_s' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'ttl_s' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._ttl_s = value
