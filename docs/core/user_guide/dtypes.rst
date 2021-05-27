======
DTypes
======

DTypes describe the different formats for recording information.

.. csv-table::
   :widths: 20, 20

   "``String``", "Saturn, a tiger, 302"
   "``Bool``", "False, True"
   "``DateTime``", "2021-04-25T08:30:32"
   "``TimeDelta``", "320D5h23m10s, 10.6s"
   "``Integer``", "0, -17, 20985"
   "``Float``", "0.034, 152.23, -13.0"

Each data type defines how data of its type can interact. In other words they
each define a set of operations that can be performed. Below is a (not
exhaustive) list of possible operations.

.. csv-table::
   :widths: 10, 30

   "``String``", "equal/not equal, order alphabetically"
   "``Bool``", "equal/not equal, greater than/less than, logical 'AND', logical 'OR'"
   "``DateTime``", "equal/not equal, greater than/less than, subtraction"
   "``TimeDelta``", "equal/not equal, greater than/less than, subtraction, addition"
   "``Integer``", "equal/not equal, greater than/less than, subtraction, addition, multiplication, division"
   "``Float``", "equal/not equal, greater than/less than, subtraction, addition, multiplication, division"

One problem with these data types is that, although you **can** perform various
operations with them, **they don't always make sense in the context** of what
the data represents.

A good example is outlined in the table below.

.. csv-table::
   :header: "Hot Sauce Name", "Category", "Cost/bottle ($)", "Spiciness (Rank)"
   :widths: 10, 10, 10, 10

   "Nando's Mild", 1, 9, 1
   "Nando's Medium", 1, 9, 4
   "Nando's Hot", 1, 9, 6
   "Tobasco Sauce", 2, 3, 2
   "Death Sauce", 3, 12, 5
   "Cholula Hot Sauce", 2, 6, 3

Although the categories are numbered, it doesn't make sense for me to say
things such as:

- the total sum of the categories is 10
- Nando's Hot spiciness is 2 times the spiciness of Cholula hot sauce
- The average category is 1.67

Statements that do make sense:

- The spiciness of Nando's hot is greater than the spiciness of Tobasco sauce
- The category of Nando's Medium is different to the category of Death sauce.
- The total cost of all the bottles is 48 ($)

And so, although useful, the DataTypes don't describe the full picture of some
recorded data. In order to distinguish between things such as Integers used
for categories, integers used for rankings, and integers used for counting the
amount of something we need a few more tools.
