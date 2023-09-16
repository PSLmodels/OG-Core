class Backpack:
    """A backpack object class. Has a name and a list of contents.
    Attributes:
    name (str): the name of the backpack's owner.
    contents (list): the contents of the backpack.
    """
    def __init__(self,name, color, max_size): #this function is the constructor.
        """Set the name and initialize an empty list of contents.
        
        Parameters:
        naem(str): the name of the backpacks owner.
        """
        self.name = name        #Initialize some attributes
        self.color = color
        self.max_size = max_size
        self.contents = []

    def put(self, item):
        if len(self.contents) >= self.max_size:
            print ("No Room!")
        else:
            self.contents.append(item)

    def take(self,item):
        self.contents.remove

    def dump(self):
        self.contents.clear



my_backpack = Backpack ("Hari", "Black", 6 )

my_backpack.put ("Notebook")
print(my_backpack.name, my_backpack.color, my_backpack.contents)



    