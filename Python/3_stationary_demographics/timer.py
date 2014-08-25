import time

pause = raw_input("Press enter to start the timer")
start = time.time()

pause2 = raw_input("Press enter to end the timer")
total =  time.time() - start

print "It took", total/60, "minutes and", total%60, "seconds"

