from dataclasses import astuple

from ..config import AddressLabels, BankLabels, PersonLabels
from ..metadata_new import Address as _Address
from ..metadata_new import Bank as _Bank
from ..metadata_new import Person as _Person


class Bank:
    def __init__(self, labels: BankLabels, name=None):
        if name is None:
            col_labels = [label for label in astuple(labels) if label is not None]
            name = "Bank_" + "_".join(col_labels)
        self._annotation = _Bank(name, labels=labels)


class Address:
    def __init__(self, labels: AddressLabels, name=None):
        if name is None:
            col_labels = [label for label in astuple(labels) if label is not None]
            name = "Address_" + "_".join(col_labels)

        self._annotation = _Address(name, labels=labels)


class Person:
    def __init__(self, labels: PersonLabels, name=None):
        if name is None:
            col_labels = [label for label in astuple(labels) if label is not None]
            name = "Person_" + "_".join(col_labels)

        self._annotation = _Person(name, labels=labels)
