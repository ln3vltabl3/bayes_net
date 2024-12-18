from itertools import combinations, count

from bnetbase import Variable, Factor, BN
import csv
import itertools

import time


def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object. 
    :return: a new Factor object resulting from normalizing factor.
    '''
    norm_scope = factor.get_scope()
    norm_sum = sum(factor.values)
    norm_vals = [x/norm_sum for x in factor.values]
    norm_f = Factor("norm_" + factor.name, norm_scope)
    norm_f.values = norm_vals
    return norm_f

def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.

    '''
    var = factor.get_variable(variable.name)
    var.set_assignment(value)
    res_scope = factor.get_scope()
    idx = res_scope.index(variable)
    res_scope.remove(variable)
    lst_dom = []
    for v in res_scope:
        lst_dom.append(v.domain())
    combo = list(itertools.product(*lst_dom))
    res_f = Factor("res_" + factor.name, res_scope)
    for var_val in combo:
        for i in range(len(var_val)):
            v = res_f.get_variable(res_scope[i].name)
            v.set_assignment(var_val[i])
        fact_val = list(var_val)[:idx] + [value] + list(var_val)[idx:]
        res_f.add_value_at_current_assignment(factor.get_value(fact_val))
    return res_f

def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''
    sum_scope = factor.get_scope()
    idx = sum_scope.index(variable)
    sum_scope.remove(variable)
    lst_dom = []
    for v in sum_scope:
        lst_dom.append(v.domain())
    combo = list(itertools.product(*lst_dom))
    sum_f = Factor("sum_" + factor.name, sum_scope)
    for var_val in combo:
        for i in range(len(var_val)):
            v = sum_f.get_variable(sum_scope[i].name)
            v.set_assignment(var_val[i])
        v_sum = 0
        for val in variable.domain():
            fact_val = list(var_val)[:idx] + [val] + list(var_val)[idx:]
            v_sum += factor.get_value(fact_val)
        sum_f.add_value_at_current_assignment(v_sum)
    return sum_f

def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors. 

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''
    if len(factor_list) == 1:
        return factor_list[0]
    elif len(factor_list) == 2:
        f_0, f_1 = factor_list[0], factor_list[1]
        mul_scope = list(set(f_0.get_scope() + f_1.get_scope()))
        lst_dom = []
        for v in mul_scope:
            lst_dom.append(v.domain())
        combo = list(itertools.product(*lst_dom))
        mul_f = Factor('mul_({},{})'.format(f_0.name, f_1.name), mul_scope)
        for var_val in combo:
            for i in range(len(var_val)):
                var = mul_scope[i]
                var.set_assignment(var_val[i])
            mul_f.add_value_at_current_assignment(f_0.get_value_at_current_assignments() *
                                                  f_1.get_value_at_current_assignments())
        return mul_f
    else:
        combo = list(combinations(factor_list,2))
        mul_c = combo[0]
        min_op = (len(mul_c[0].values) * len(mul_c[1].values)
                  if not(set(mul_c[0].get_scope()) & set(mul_c[1].get_scope()))
                  else max(len(mul_c[0].values), len(mul_c[1].values)))
        for c in combo[1:]:
            f_0, f_1 = c[0], c[1]
            num_op = (len(f_0.values) * len(f_1.values) if not(set(f_0.get_scope()) & set(f_1.get_scope()))
                       else max(len(f_0.values), len(f_1.values)))
            if min_op > num_op:
                mul_c = c
                min_op = num_op
        factor_list.remove(mul_c[0])
        factor_list.remove(mul_c[1])
        mul_f = multiply(mul_c)
        factor_list.append(mul_f)
        return multiply(factor_list)



def ve(bayes_net, var_query, EvidenceVars):
    '''

    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the 
    evidence provided by EvidenceVars. 

    :param bayes_net: a BN object.
    :param var_query: the query variable. we want to compute a distribution
                     over the values of the query variable.
    :param EvidenceVars: the evidence variables. Each evidence variable has 
                         its evidence set to a value from its domain 
                         using set_evidence.
    :return: a Factor object representing a distribution over the values
             of var_query. that is a list of numbers, one for every value
             in var_query's domain. These numbers sum to 1. The i-th number
             is the probability that var_query is equal to its i-th value given 
             the settings of the evidence variables.

    For example, assume that
        var_query = A with Dom[A] = ['a', 'b', 'c'], 
        EvidenceVars = [B, C], and 
        we have called B.set_evidence(1) and C.set_evidence('c'), 
    then VE would return a list of three numbers, e.g. [0.5, 0.24, 0.26]. 
    These numbers would mean that 
        Pr(A='a'|B=1, C='c') = 0.5, 
        Pr(A='b'|B=1, C='c') = 0.24, and
        Pr(A='c'|B=1, C='c') = 0.26.

    '''
    factor_list = bayes_net.factors()
    for factor in bayes_net.factors():
        ev_vars = list(set(factor.get_scope()) & set(EvidenceVars))
        prev_f = factor
        for ev_var in ev_vars:
            ev_val = ev_var.get_evidence()
            res_f = restrict(prev_f, ev_var, ev_val)
            factor_list.remove(prev_f)
            factor_list.append(res_f)
            prev_f = res_f
    hidden_factors = []
    for i in range(len(factor_list)-1, -1, -1):
        for var in factor_list[i].get_scope():
            if not(var == var_query or var in EvidenceVars):
                hidden_factors.append(factor_list.pop(i))
                break
    hid_f = multiply(hidden_factors)
    for var in hid_f.get_scope():
        if not(var == var_query or var in EvidenceVars):
            hid_f = sum_out(hid_f, var)
    factor_list.append(hid_f)
    return normalize(multiply(factor_list))


def naive_bayes_model(data_file, variable_domains = {"Work": ['Not Working', 'Government', 'Private', 'Self-emp'], "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'], "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'], "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'], "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'], "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], "Gender": ['Male', 'Female'], "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'], "Salary": ['<50K', '>=50K']}, class_var = Variable("Salary", ['<50K', '>=50K'])):
    '''
   NaiveBayesModel returns a BN that is a Naive Bayes model that 
   represents the joint distribution of value assignments to 
   variables in the Adult Dataset from UCI.  Remember a Naive Bayes model
   assumes P(X1, X2,.... XN, Class) can be represented as 
   P(X1|Class)*P(X2|Class)* .... *P(XN|Class)*P(Class).
   When you generated your Bayes bayes_net, assume that the values 
   in the SALARY column of the dataset are the CLASS that we want to predict.
   @return a BN that is a Naive Bayes model and which represents the Adult Dataset. 
    '''
    ### READ IN THE DATA
    input_data = []
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)

    ### DOMAIN INFORMATION REFLECTS ORDER OF COLUMNS IN THE DATA SET
    #variable_domains = {
    #"Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
    #"Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
    #"Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
    #"MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
    #"Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    #"Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    #"Gender": ['Male', 'Female'],
    #"Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
    #"Salary": ['<50K', '>=50K']
    #}
    counter_dict = {}
    var_list = []
    factor_list = []
    for key in variable_domains.keys():
        if key != class_var.name:
            v = Variable(key, variable_domains[key])
            var_list.append(v)
            factor_list.append(Factor("P("+key+"|"+class_var.name+")", [v, class_var]))
            counter_dict[v] = [([0] * class_var.domain_size())[:] for _ in range(v.domain_size())]
        else:
            var_list.append(class_var)
            factor_list.append(Factor("P("+class_var.name+")", [class_var]))
            counter_dict[class_var] = [0] * class_var.domain_size()
    c_var_idx = var_list.index(class_var)
    for val_list in input_data:
        for i in range(len(val_list)):
            class_idx = class_var.domain().index(val_list[c_var_idx])
            if i != c_var_idx:
                var = var_list[i]
                val = val_list[i]
                val_idx = variable_domains[var.name].index(val)
                counter_dict[var][val_idx][class_idx] += 1
            else:
                counter_dict[class_var][class_idx] += 1
    class_sum = counter_dict[class_var]
    for i in range(len(var_list)):
        var = var_list[i]
        if var != class_var:
            for j in range(var.domain_size()):
                counter_dict[var][j][:] = [x/y for x,y in zip(counter_dict[var][j], class_sum)]
                var.set_assignment(var.domain()[j])
                for k in range(class_var.domain_size()):
                    class_var.set_assignment(class_var.domain()[k])
                    factor_list[i].add_value_at_current_assignment(counter_dict[var][j][k])
        else:
            counter_dict[class_var][:] = [x/sum(class_sum) for x in counter_dict[class_var]]
            for j in range(class_var.domain_size()):
                class_var.set_assignment(class_var.domain()[j])
                factor_list[i].add_value_at_current_assignment(counter_dict[class_var][j])
    return BN(data_file + "_BN", var_list, factor_list)


def explore(bayes_net, question):
    '''    Input: bayes_net---a BN object (a Bayes bayes_net)
           question---an integer indicating the question in HW4 to be calculated. Options are:
           1. What percentage of the women in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
           2. What percentage of the men in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
           3. What percentage of the women in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
           4. What percentage of the men in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
           5. What percentage of the women in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
           6. What percentage of the men in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
           @return a percentage (between 0 and 100)
    '''

    ### READ IN THE DATA
    input_data = []
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)  # skip header row
        for row in reader:
            input_data.append(row)

    ev_indices = [0,1,3,4]
    ev_var = []
    for idx in ev_indices:
        ev_var.append(bayes_net.get_variable(headers[idx]))
    gender = 'Female' if question % 2 else 'Male'
    numerator = 0
    denominator = 0
    query_var = bayes_net.get_variable(headers[8])
    query_var.set_assignment('>=50K')
    for val_list in input_data:
        for idx in ev_indices:
            bayes_net.get_variable(headers[idx]).set_evidence(val_list[idx])
        pr_given_ev = ve(bayes_net, query_var , ev_var)
        if gender == val_list[6]:
            if (question - 1 )// 2 == 0:
                bayes_net.get_variable(headers[6]).set_evidence(val_list[6])
                pr_given_ev2 = ve(bayes_net, query_var, list(ev_var) + [bayes_net.get_variable(headers[6])])
                if pr_given_ev.get_value_at_current_assignments() > pr_given_ev2.get_value_at_current_assignments():
                    numerator += 1
                denominator += 1
            elif (question - 1) // 2 == 1:
                if pr_given_ev.get_value_at_current_assignments() > 0.5:
                    if val_list[-1] == '>=50K':
                        numerator += 1
                    denominator += 1
            else:
                if pr_given_ev.get_value_at_current_assignments() > 0.5:
                    numerator += 1
                denominator += 1
    return numerator/denominator * 100

if __name__ == '__main__':
    nb = naive_bayes_model('data/adult-train.csv')
    for i in range(1,7):
        start_time = time.time()
        print("explore(nb,{}) = {}".format(i, explore(nb, i)))
        print("--- %s seconds ---" % (time.time() - start_time))