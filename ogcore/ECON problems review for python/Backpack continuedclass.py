from Classobjects import Backpack
 #storing varibales as attributes, classes can have functions attatched to them. A function that belogns to a specific class is called a method.

class Backpack:
    #...
    def put(self, item):
       # """Add 'item' to the backpacks list of contents."""
       self.contents.append(item) # use self.contents not just cotents
    
    def take(self, item):
        self.contents.remove(item)

# self argument is only included in the declaration of the class methods, not when calling the methods on an instantiation of the class.""  

#add some items to the backpack object
my_backpack = Backpack ("Fred")
type(my_backpack)


#Access the objects attributes with a period and the attribute name
print(my_backpack.name, my_backpack.contents)
 #EVery object in python has built in attributes. For example,modules have a __name__ attribute that identifies the scope in which it is being executed.

 #storing varibales as attributes, classes can have functions attatched to them. A function that belogns to a specific class is called a method.

my_backpack.put ("notebook") #my_backpack is passed implicitly to backpack.put as the first argument.
my_backpack.put ("pencils")
 
