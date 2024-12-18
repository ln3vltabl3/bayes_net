#     C) class BN<br>
#        This class allows one to put factors and variables together to form a Bayes net.<br>
#        It serves as a convient place to store all of the factors and variables associated<br>
#        with a Bayes Net in one place. It also has some utility routines to, e.g,., find<br>
#        all of the factors a variable is involved in. <br>
#   
# 

class Variable:
    '''
    Class for defining a variable.
    When creating the variable, you need to provide a name and its domain
    as a list. You can add more domain value after creating the variable.

    If this variable is an evidence variable, it keeps track of 
    the index of the observed value in its domain using evidence_index.

    When this variable is used in a Factor, it keeps track of 
    the index of the assigned value in its domain using assignment_index.
    '''

    def __init__(self, name, domain=[]):
        '''
        Create a variable object, specifying its name (a string). 
        Optionally specify the initial domain.
        :param name: name of the variable as a string
        :param domain: a list of domain values. 
        '''
        self.name = name                
        self.dom = list(domain)         

        # If this variable is an evidence variable
        # This is the index of the observed value in its domain.
        self.evidence_index = 0         

        # For use by factors. We can assign variables values
        # and these assigned values can be used by factors
        # to index into their tables.
        self.assignment_index = 0       

    def add_domain_values(self, values):
        '''
        Add values to the domain.
        :param values: a list of values to be added to the domain
        '''
        for val in values: 
            self.dom.append(val)

    def value_index(self, value):
        '''
        Return the index of the given value in the domain.
        :param value: the value to look up in the domain
        :return index of the given value in the domain
        '''
        return self.dom.index(value)

    def domain_size(self):
        '''
        Return the size of the domain
        :return the size of the domain
        '''
        return len(self.dom)

    def domain(self):
        '''
        Return the domain as a list
        :return the domain as a list
        '''
        return list(self.dom)

    # If this variable is an evidence variable,
    # use the following two functions to set and get its observed value
    def set_evidence(self, val):
        '''
        When this variable is an evidence variable,
        Set evidence_index to be the index of the given value in the domain.
        :param val: observed value of the variable
        '''
        self.evidence_index = self.value_index(val)

    def get_evidence(self):
        '''
        When this variable is an evidence variable,
        Return the value of the variable.
        :return the observed value of this variable
        '''
        return self.dom[self.evidence_index]


    def set_assignment(self, val):
        '''
        Set this variable's assignment value for factor lookups
        '''
        self.assignment_index = self.value_index(val)

    def get_assignment(self):
        '''
        '''
        return self.dom[self.assignment_index]

    ## Special low-level routines used directly by the Factor object
    def set_assignment_index(self, index):
        '''This routine is used by the factor objects'''
        self.assignment_index = index

    def get_assignment_index(self):
        '''This routine is used by the factor objects'''
        return(self.assignment_index)

    def __repr__(self):
        '''string to return when evaluating the object'''
        return("{}".format(self.name))
    
    def __str__(self):
        '''more elaborate string for printing'''
        return("{}, Dom = {}".format(self.name, self.dom))


class Factor: 
    '''
    Class for defining a factor.

    The scope of a factor contains an ordered list of variables.
    A factor maps every assignment of values to these variables to a number. 
    If the scope is empty, the factor contains a single number.

    After creating the factor, you can interact with the factor in multiple ways.

    If you interact with the factor using add_values and get_value, 
    then you must provide the list of values in order of the variables
    in the factor's scope.

    If you interact with the factor using 
    add_value_at_current_assignment and get_value_at_current_assignments,
    you must assign the variables using set_assignment, then you can
    set or get the value of the factor. 

    The advantage of the latter approach is that there is no need
    to worry about the order of the variables in the factor. 
    The disadvantage of the latter approach is that you may have to 
    save and restore the values of the variables. 

    '''
    def __init__(self, name, scope):
        '''
        Create a Factor object.

        :param name: the name of the factor as a string.
        :param scope: an ordered list of Variables in the factor.
        '''
        self.name = name
        self.scope = list(scope)

        size = 1
        for v in scope:
            size = size * v.domain_size()
        self.values = [0] * size  #initialize values to be a list of zeros.

    def get_scope(self):
        '''
        Return a copy of the scope so that we can modify the 
        returned copy without affecting this factor
        :return a copy of the scope
        '''
        return list(self.scope)

    def get_variable(self, name):
        '''
        Return the variable with the given name
        '''
        for v in list(self.scope):
            if v.name == name:
                return v
        return None        

    def add_values(self, values):
        '''
        We can use this function to initialize the factor. 
        
        values is a list of list of values.
        Each inner list contains one value for each variable followed by a number.
        The values for the variables are ordered according to the factor's scope.
        The last number is the factor's value when its variables are assigned
        these values.
        
        :param values: a list of list of values.
                       
        For example, if 
        scope = [A, B, C], 
        A.domain() = [1, 2, 3], 
        B.domain() = ['a', 'b'], and 
        C.domain() = ['heavy', 'light'], 
        then we could initialize the factor by passing it the following list:

        [[1, 'a', 'heavy', 0.25], [1, 'a', 'light', 1.90],
         [1, 'b', 'heavy', 0.50], [1, 'b', 'light', 0.80],
         [2, 'a', 'heavy', 0.75], [2, 'a', 'light', 0.45],
         [2, 'b', 'heavy', 0.99], [2, 'b', 'light', 2.25],
         [3, 'a', 'heavy', 0.90], [3, 'a', 'light', 0.111],
         [3, 'b', 'heavy', 0.01], [3, 'b', 'light', 0.1]]

        Afterwards, the factor's values will be initialized accordingly.
        For example, the value of (A=1,B='b',C='light') is 0.80.
         '''
        for t in values:
            index = 0
            for v in self.scope:
                index = index * v.domain_size() + v.value_index(t[0])
                t = t[1:]
            self.values[index] = t[0]
         

    def get_value(self, variable_values):
        '''
        This function is used to retrieve a value from the factor. 
        We pass it an ordered list of values, one for every
        variable in self.scope. It then returns the factor's value on
        that set of assignments.  For example, if self.scope = [A, B,
        C], and A.domain() = [1,2,3], B.domain() = ['a', 'b'], and
        C.domain() = ['heavy', 'light'], and we invoke this function
        on the list [1, 'b', 'heavy'] we would get a return value
        equal to the value of this factor on the assignment (A=1,
        B='b', C='light')
        '''
        index = 0
        for v in self.scope:
            index = index * v.domain_size() + v.value_index(variable_values[0])
            variable_values = variable_values[1:]
        return self.values[index]


    def add_value_at_current_assignment(self, number): 
        '''This function allows adding values to the factor in a way
        that will often be more convenient. We pass it only a single
        number. It then looks at the assigned values of the variables
        in its scope and initializes the factor to have value equal to
        number on the current assignment of its variables. Hence, to
        use this function one first must set the current values of the
        variables in its scope.

        For example, if 
            scope = [A, B, C],
            A.domain() = [1,2,3], 
            B.domain() = ['a', 'b'], and
            C.domain() = ['heavy', 'light'], 
        and we first set an assignment for A, B and C:
            A.set_assignment(1)
            B.set_assignment('a')
            C.set_assignment('heavy')
        then we call add_value_at_current_assignment(0.33) with the value 0.33, 
        we would have initialized this factor to have the value 0.33 
        on the assigments (A=1, B='1', C='heavy')
        This has the same effect as the call 
            add_values([1, 'a', 'heavy', 0.33])

        One advantage of the current_assignment interface to factor values is that
        we don't have to worry about the order of the variables in the factor's
        scope. add_values on the other hand has to be given tuples of values where 
        the values must be given in the same order as the variables in the factor's 
        scope. 

        See recursive_print_values called by print_table to see an example of 
        where the current_assignment interface to the factor values comes in handy.
        '''
        index = 0
        for v in self.scope:
            index = index * v.domain_size() + v.get_assignment_index()
        self.values[index] = number


    def get_value_at_current_assignments(self):
        '''
        Retrieve a value from the factor. 
        The value retrieved is the value of the factor when
        evaluated at the current assignment to the variables in its
        scope.

        For example, if self.scope = [A, B, C], and A.domain() =
        [1,2,3], B.domain() = ['a', 'b'], and C.domain() = ['heavy',
        'light'], and we had previously invoked A.set_assignment(1),
        B.set_assignment('a') and C.set_assignment('heavy'), then this
        function would return the value of the factor on the
        assigments (A=1, B='1', C='heavy')
        '''
        
        index = 0
        for v in self.scope:
            index = index * v.domain_size() + v.get_assignment_index()
        return self.values[index]

    def print_table(self):
        '''
        Print the factor's table
        '''
        saved_values = []  #save and then restore the variable assigned values.

        for v in self.scope:
            saved_values.append(v.get_assignment_index())

        self.recursive_print_values(self.scope)

        for v in self.scope:
            v.set_assignment_index(saved_values[0])
            saved_values = saved_values[1:]
        
    def recursive_print_values(self, vars):
        '''

        '''
        if len(vars) == 0:
            print("[",end=""),
            for v in self.scope:
                print("{} = {},".format(v.name, v.get_assignment()), end="")
            print("] = {}".format(self.get_value_at_current_assignments()))
        else:
            for val in vars[0].domain():
                vars[0].set_assignment(val)
                self.recursive_print_values(vars[1:])

    def __repr__(self):
        return("{}".format(self.name))



class BN:
    '''Class for defining a Bayes Net.
       This class is simple, it just is a wrapper for a list of factors. And it also
       keeps track of all variables in the scopes of these factors'''
    def __init__(self, name, Vars, Factors):
        self.name = name
        self.Variables = list(Vars)
        self.Factors = list(Factors)
        for f in self.Factors:
            for v in f.get_scope():     
                if not v in self.Variables:
                    print("Bayes net initialization error")
                    print("Factor scope {} has variable {} that", end='')
                    print(" does not appear in list of variables {}.".format(list(map(lambda x: x.name, f.get_scope())), v.name, list(map(lambda x: x.name, Vars))))

    def factors(self):
        '''
        Return the list of factors in the Bayes Net.
        '''
        return list(self.Factors)

    def variables(self):
        '''
        Return the list of variables in the Bayes Net.
        '''
        return list(self.Variables)

    def get_variable(self, name):
        '''
        Return the variable with the given name
        '''
        for v in list(self.Variables):
            if v.name == name:
                return v
        return None        

