# Greedy 1-to-m matching algorithm
# The idea is that we match students according to their preferencs
# The ordering of the students is important: we pay their preferences more attention.
# If a student is unmatched and their preferences are empty, we assign them to a project that already has students, but also has available places.  The special project is exempt from this rule.
# If 3 students each have diffeernt preferences and prefer 1>2>3, 2>3>1, 3>1>2, then somebody will always be assigned their 1st choice, somebody their 2nd and somebody their 3rd. 


import queue

studentPref = {
    "viola":        [5, 6], 
    "isabelle":     [5, 6], 
    "marta":        [5, 6], 
    "allizha":      [6, 5, 4],
    "boris":        [6, 5, 4],
    "rithika":      [6, 5, 4],
    "jing":         [4, 2, 5],
    "mengmeng":     [4, 2, 5],  
    "ebrima":       [4, 2, 5], 
    "bill":          [2, 1, 6],
    "mateo":        [2, 1, 6], 
    "felix":        [2, 1, 6], 
    "shao tong":    [3, 2, 4, 1],
    "yi qi":        [3, 2, 4, 1],
    "bo le":        [3, 2, 4, 1],
    "guodong":      [1, 2, 6], 
    "lishang":      [1, 2, 6], 
    "pengcheng":    [1, 2, 6], 
    "alec":         [1, 3, 4], 
    "aurelie":      [1, 3, 4], 
    "sai":          [3, 1, 4,],
    "sampson":      [4, 1,2,], 
    "daksh":        [1, 3,], 
    "songzhi":      [1, 4, 6] 
}

groups = [["viola", "isabelle", "marta"],
          ["bill", "mateo", "felix"],
          ["allizha", "boris", "rithika"],
          ["guodong", "lishang", "pengcheng"],
          ["jing", "mengmeng", "ebrima"],
          ["alec", "aurelie"],
          ["shao tong", "yi qi", "bo le"],
          ["sai"],
          ["sampson"],
          ["daksh"],
          ["songzhi"]]

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
                        if (places[k] > 0):
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
    for i in range(len(groups)):
        print ("G ", i)
        for s in groups[i]:
            print (s, assignment[s])
            
if __name__ == "__main__":
    main()
