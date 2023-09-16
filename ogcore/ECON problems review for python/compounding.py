
def calcinterest(principal, interest_rate, time_period, frequency):
    amount = principal * (1 + (interest_rate / compounding_frequency)) ** (compounding_frequency* time_period)
    interestamt = amount - principal
    return interestamt

#get user input
principal = float(input("Enter the principal Amount"))
interest_rate = float(input("Enter the interest rate as a decimal: "))
time_period = float(input("enter the time period: "))
compounding_frequency = float(input("Enter the compunding frequency in years: "))

#print out
totalInterest = calcinterest(principal, interest_rate, time_period, compounding_frequency)
print("Total interest accrued: $", round(totalInterest))