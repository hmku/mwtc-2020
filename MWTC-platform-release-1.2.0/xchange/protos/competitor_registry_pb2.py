# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: competitor_registry.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import competitor_pb2 as competitor__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='competitor_registry.proto',
  package='edu.uchicago.midwesttrading.xchange',
  syntax='proto3',
  serialized_options=b'B\022CompRegistryProtos',
  serialized_pb=b'\n\x19\x63ompetitor_registry.proto\x12#edu.uchicago.midwesttrading.xchange\x1a\x10\x63ompetitor.proto\"k\n\x19RegisterCompetitorRequest\x12N\n\ncompetitor\x18\x01 \x01(\x0b\x32:.edu.uchicago.midwesttrading.xchange.CompetitorCredentials\"\xc1\x01\n\x1aRegisterCompetitorResponse\x12V\n\x06status\x18\x01 \x01(\x0e\x32\x46.edu.uchicago.midwesttrading.xchange.RegisterCompetitorResponse.Status\x12\x0f\n\x07message\x18\x02 \x01(\t\":\n\x06Status\x12\x0b\n\x07SUCCESS\x10\x00\x12\x16\n\x12\x41LREADY_REGISTERED\x10\x01\x12\x0b\n\x07\x46\x41ILURE\x10\x02\"g\n\x15\x41waitCaseStartRequest\x12N\n\ncompetitor\x18\x01 \x01(\x0b\x32:.edu.uchicago.midwesttrading.xchange.CompetitorCredentials\"\x18\n\x16\x41waitCaseStartResponseB\x14\x42\x12\x43ompRegistryProtosb\x06proto3'
  ,
  dependencies=[competitor__pb2.DESCRIPTOR,])



_REGISTERCOMPETITORRESPONSE_STATUS = _descriptor.EnumDescriptor(
  name='Status',
  full_name='edu.uchicago.midwesttrading.xchange.RegisterCompetitorResponse.Status',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SUCCESS', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALREADY_REGISTERED', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FAILURE', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=329,
  serialized_end=387,
)
_sym_db.RegisterEnumDescriptor(_REGISTERCOMPETITORRESPONSE_STATUS)


_REGISTERCOMPETITORREQUEST = _descriptor.Descriptor(
  name='RegisterCompetitorRequest',
  full_name='edu.uchicago.midwesttrading.xchange.RegisterCompetitorRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='competitor', full_name='edu.uchicago.midwesttrading.xchange.RegisterCompetitorRequest.competitor', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=84,
  serialized_end=191,
)


_REGISTERCOMPETITORRESPONSE = _descriptor.Descriptor(
  name='RegisterCompetitorResponse',
  full_name='edu.uchicago.midwesttrading.xchange.RegisterCompetitorResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='edu.uchicago.midwesttrading.xchange.RegisterCompetitorResponse.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message', full_name='edu.uchicago.midwesttrading.xchange.RegisterCompetitorResponse.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _REGISTERCOMPETITORRESPONSE_STATUS,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=194,
  serialized_end=387,
)


_AWAITCASESTARTREQUEST = _descriptor.Descriptor(
  name='AwaitCaseStartRequest',
  full_name='edu.uchicago.midwesttrading.xchange.AwaitCaseStartRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='competitor', full_name='edu.uchicago.midwesttrading.xchange.AwaitCaseStartRequest.competitor', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=389,
  serialized_end=492,
)


_AWAITCASESTARTRESPONSE = _descriptor.Descriptor(
  name='AwaitCaseStartResponse',
  full_name='edu.uchicago.midwesttrading.xchange.AwaitCaseStartResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=494,
  serialized_end=518,
)

_REGISTERCOMPETITORREQUEST.fields_by_name['competitor'].message_type = competitor__pb2._COMPETITORCREDENTIALS
_REGISTERCOMPETITORRESPONSE.fields_by_name['status'].enum_type = _REGISTERCOMPETITORRESPONSE_STATUS
_REGISTERCOMPETITORRESPONSE_STATUS.containing_type = _REGISTERCOMPETITORRESPONSE
_AWAITCASESTARTREQUEST.fields_by_name['competitor'].message_type = competitor__pb2._COMPETITORCREDENTIALS
DESCRIPTOR.message_types_by_name['RegisterCompetitorRequest'] = _REGISTERCOMPETITORREQUEST
DESCRIPTOR.message_types_by_name['RegisterCompetitorResponse'] = _REGISTERCOMPETITORRESPONSE
DESCRIPTOR.message_types_by_name['AwaitCaseStartRequest'] = _AWAITCASESTARTREQUEST
DESCRIPTOR.message_types_by_name['AwaitCaseStartResponse'] = _AWAITCASESTARTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RegisterCompetitorRequest = _reflection.GeneratedProtocolMessageType('RegisterCompetitorRequest', (_message.Message,), {
  'DESCRIPTOR' : _REGISTERCOMPETITORREQUEST,
  '__module__' : 'competitor_registry_pb2'
  # @@protoc_insertion_point(class_scope:edu.uchicago.midwesttrading.xchange.RegisterCompetitorRequest)
  })
_sym_db.RegisterMessage(RegisterCompetitorRequest)

RegisterCompetitorResponse = _reflection.GeneratedProtocolMessageType('RegisterCompetitorResponse', (_message.Message,), {
  'DESCRIPTOR' : _REGISTERCOMPETITORRESPONSE,
  '__module__' : 'competitor_registry_pb2'
  # @@protoc_insertion_point(class_scope:edu.uchicago.midwesttrading.xchange.RegisterCompetitorResponse)
  })
_sym_db.RegisterMessage(RegisterCompetitorResponse)

AwaitCaseStartRequest = _reflection.GeneratedProtocolMessageType('AwaitCaseStartRequest', (_message.Message,), {
  'DESCRIPTOR' : _AWAITCASESTARTREQUEST,
  '__module__' : 'competitor_registry_pb2'
  # @@protoc_insertion_point(class_scope:edu.uchicago.midwesttrading.xchange.AwaitCaseStartRequest)
  })
_sym_db.RegisterMessage(AwaitCaseStartRequest)

AwaitCaseStartResponse = _reflection.GeneratedProtocolMessageType('AwaitCaseStartResponse', (_message.Message,), {
  'DESCRIPTOR' : _AWAITCASESTARTRESPONSE,
  '__module__' : 'competitor_registry_pb2'
  # @@protoc_insertion_point(class_scope:edu.uchicago.midwesttrading.xchange.AwaitCaseStartResponse)
  })
_sym_db.RegisterMessage(AwaitCaseStartResponse)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
