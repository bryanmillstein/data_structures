import unittest

from practice import VendingMachine


class TestVendingMachineMethods(unittest.TestCase):

    def testAddItem(self):
        machine = VendingMachine()
        item = machine.addItem('chips')

        self.assertEqual(machine.count, 1)
        self.assertEqual(machine.items['A' + str(machine.count)], item.name)


unittest.main()