# Greedy 1-to-m matching algorithm
# The idea is that we match students according to their preferencs
# The ordering of the students is important: we pay their preferences more attention.
# If a student is unmatched and their preferences are empty, we assign them to a project that already has students, but also has available places
# Arrow's theorem applies here: if 3 students each have diffeernt preferences and are alone, so that they are treated as a group, and prefer 1>2>3, 2>3>1, 3>1>2, then somebody will always be assigned their 1st choice, somebody their 2nd and somebody their 3rd. This is because of the 'fill projects' rule. The special project is exempt from this rule


import queue

studentPref = {
    "viola":        [1, 5, 6], #5
    "isabelle":     [1, 5, 6], #4.5
    "marta":        [1, 5, 6], #5
    "alec":         [1, 3, 4], # 5.5 #g7
    "aurelie":      [1, 3, 4], # 5 #g7
    "allizha":      [6, 5, 4], #5
    "boris":        [6, 5, 4], #5
    "rithika":      [6, 5, 4], #4.5
    "jing":         [4, 2, 5] ,#5
    "mengmeng":     [4, 2, 5], #5 
    "ebrima":       [4, 2, 5], #4.5
    "bil":          [2, 1, 6], #4
    "mateo":        [2, 1, 6], #4.5
    "felix":        [2, 1, 6], #5
    "shao tong":    [3, 2, 4, 1], # 4
    "yi qi":        [3, 2, 4, 1], #5
    "bo le":        [3, 2, 4, 1], #4
    "guodong":      [5, 1, 2], #3.5
    "lishang":      [5, 1, 2], #5
    "pengcheng":    [5, 1, 2], #4.5
    "sai":          [3, 1, 4], #5
    "sampson":      [4, 1], # 4.5 g11
    "daksh":        [1, 3], #3.5,
    "songzhi":      [1, 4, 6] # 3 # g12
}

import numpy as np

def main():
    students = list(studentPref.keys())
    n_projects = 7
    n_students = len(students)
    assignment = {x : -1 for x in students}
    places = 4 + np.zeros(n_projects) 
    ## assign student to their first choice initially, and then go down
    priority = 0

    while True:
        n_assigned = 0
        for s in students:
            if assignment[s] >= 0:
                n_assigned += 1
            else:
                if (priority < len(studentPref[s])):
                    project = studentPref[s][priority]
                else:
                    project = -1
                    tmp_places = 4
                    for k in range(n_projects):
                        if (places[k] > 0 and k >0):
                            if places[k] < tmp_places:
                                project = k
                                tmp_places = places[k]
                                
                    print("Going for project ", project)
                if (places[project] > 0):
                    assignment[s] = project
                    places[project] -=1
                    n_assigned +=1
        print(assignment)
        print(places)
        print("assigned: ", n_assigned)
        if n_assigned == n_students:
            print("All students assigned")
            break
        priority += 1
        
    print(assignment)
    print(places)
if __name__ == "__main__":
    main()
