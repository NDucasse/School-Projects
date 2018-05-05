Your Lisp program should read in family relationships from a file given on the command line. Each file
entry contains a list of information about a person: name, gender, parents, and children. For example, if
Bob (male) has children named Matt and Becky, and parents named Sam and Doris, then the
corresponding file entry would be:
(bob male (sam doris) (matt becky))
Use Lisp structs to create and access the genealogy database. The only fields allowed are name, gender,
parents, and children. The name field stores the person’s name; the gender field is either male or female.
The parents and children fields are lists consisting of zero or more names.
(defstruct person name gender parents children)
Your program should include a function which reads information from the family tree file, builds the
database, and stores it in a global variable called *database*. This function should be called
automatically when the program is run from the command line. The database may then be queried by
typing requests such as
> (children 'bob)
(matt becky) ; Lisp response
A database query returns a list of zero or more names that correspond to the specified family
relationship.