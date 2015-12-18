class Fraction:

	def __init__(self, top, bottom):
		self.numerator = top
		self.denominator = bottom

	def __str__(self):
		return "{0}/{1}".format(self.numerator, self.denominator)

	def __add__(self, second_fraction):

		new_numerator = self.numerator*second_fraction.denominator + second_fraction.numerator*self.denominator
		new_denominator = self.denominator*second_fraction.denominator

		return self.lowest_terms(new_numerator, new_denominator) 

	def lowest_terms(self, numerator, denominator):
		if numerator < denominator:
			check = numerator
		else:
			check = denominator

		while check > 0:
			if numerator % check == 0 and denominator % check == 0:
				numerator = numerator / check
				denominator = denominator / check
			check = check - 1

		return numerator,denominator


myFraction = Fraction(13,10)
myFraction2 = Fraction(2,5)
print myFraction.__add__(myFraction2)
