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

