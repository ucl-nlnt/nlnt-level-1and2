import random

class prompt_randomizer:
  #def __init__(self):
  #  self.init = [0.0, 0.0]
  
  def rand_dist ():
    meas = ["mm", " millimeters", "cm", " centimeters", "m", " meters", '"', "in", " inches", "'", "ft", " feet", "yds", " yards"]
    dis_type = random.choice(meas)
    #dis_type = meas[1]

    written_num = ["actual number", "quarters", "spelled out"]
    selected_num_type = random.choice(written_num)

    if selected_num_type == "actual number":
      if dis_type == "millimeters" or dis_type == "mm":
        rand_num = float(random.randint (1000, 300000))/100
        in_meters = rand_num / 1000
      elif dis_type == "cm" or dis_type == "centimeters":
        rand_num = float(random.randint (100, 3000))/100
        in_meters = rand_num / 100
      elif dis_type == "m" or dis_type == "meters":
        rand_num = float(random.randint (1, 300))/100
        in_meters = rand_num
      elif dis_type == '"' or dis_type == "in" or dis_type == "inches":
        rand_num = float(random.randint (39, 11811))/100
        in_meters = round(rand_num/39.37, 2)
      elif dis_type == "'" or dis_type == "ft" or dis_type == "feet":
        rand_num = float(random.randint (3, 984))/100
        in_meters = round(rand_num/3.281, 2)
      elif dis_type == "yds" or dis_type == "yards":
        rand_num = float(random.randint (1, 328))/100
        in_meters = round(rand_num/1.094, 2)
      else:
        rand_num = 1

      if rand_num == 1:
        meas = ["m", " meter", "ft", " foot", " ruler", " yard", " yardstick"]
        mets = [1, 1, 0.31, 0.31, 0.31, 0.91, 0.91]
        num = random.randint(0, len(meas)-1)
        dis_type = meas[num]
        in_meters = mets[num]

      #space = ['', ' ']
      #return (str(rand_num) + random.choice(space) + dis_type, in_meters)
      return (str(rand_num) + dis_type, in_meters)

    elif selected_num_type == "quarters":
      rand_quarter = ["a single", "a quarter", "a third", "a fourth", "a quarter", "a fifth", "a half", "three quarters", "an eigth", "two fifths", "three eighths", "five eights"]
      writ = [1, 0.25, 0.30, 0.25, 0.25, 0.20, 0.5, 0.75, 0.125, 0.40, 0.375, 0.625]
      measure = [" meter", " foot", " ruler", " yardstick"]
      num = random.randint(0, len(rand_quarter)-1)

      meas = random.randint(0, len(measure)-1)

      if meas == 0:
        in_meters = writ[num]
      elif meas == 1 or meas == 2:
        in_meters = round(writ[num]/3.281, 2)
      elif meas == 3:
        in_meters = round(writ[num]/1.094, 2)

      if num == 1:
        return (rand_quarter[num] + " " + measure[meas], in_meters)
      else:
        return (rand_quarter[num] + " of a" + measure[meas], in_meters)

    elif selected_num_type == "spelled out":
      prepositions_b = ["by", "a distance of", "for", "for a total distance of"]
      numbers = ["a", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven","twelve","thirteen","fourteen","fifteen","sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
      writ = [1,1,2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
      added = ["", "full ", "whole "]

      meas = random.randint(0, 4)

      if meas == 0:
        num = random.randint(0, len(numbers)-1)
        in_meters = writ[num] / 100
      elif meas == 1:
        num = random.randint(0, 3)
        in_meters = writ[num]
      elif meas == 2 or meas == 3:
        num = random.randint(0, 9)
        in_meters = round(writ[num]/3.281, 2)
      elif meas == 4:
        num = random.randint(0, 3)
        in_meters = round(writ[num]/1.094, 2)

      if num == 0 or num == 1:
        measure = ["centimeter", "meter", "foot", "ruler", "yardstick"]
      else:
        measure = ["centimeters", "meters", "feet", "rulers", "yardsticks"]

      return (numbers[num] + " " + random.choice(added) + measure[meas], in_meters)

  def rand_multiples(max_left):
    multiples = ["", "twice", "two times", "2 times", "2x", "thrice", "three times", "3x", "3 times", "4 times", "4x", "four times", "five times", "5x", "5 times"]
    times = [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    
    if max_left == 0:
    	time_int = 0
    elif max_left == 1:
    	time_int = random.randint(0, 4)
    elif max_left == 2:
    	time_int = random.randint(0, 8)
    elif max_left == 3:
    	time_int = random.randint(0, 11)
    else:
    	time_int = random.randint(0, len(multiples)-1)
    
    return (multiples[time_int], times[time_int])

  #def rand_insts(max_per_prompt):
  #  no_of_instructions = random.randint(1, max_per_prompt)
  #  return no_of_instructions

  def rand_rot():
    angle_type = ["degrees", "radians", "rads", "degs", "%", "fraction"]
    rot_type = random.choice(angle_type)

    if rot_type == "degrees" or rot_type == "degs":
      deg = random.randint(0, 360)
      return (str(deg) + " " + rot_type, deg)

    elif rot_type == "radians" or rot_type == "rads":
      rad_type = ["numerical", "pi"]
      a = random.choice(rad_type)

      if a == "numerical":
        rad = float(random.randint(0, 628))/100
        deg = round(rad * 57.296, 2)
        return ( str(rad) + " " + rot_type, deg)

      elif a == "pi":
        pi_rot = ["pi/6", "pi/4", "pi/3", "pi/2", "pi", "2pi/3", "3pi/4", "5pi/6", "7pi/6", "5pi/4", "4pi/3", "3pi/2", "5pi/3", "7pi/4", "11pi/6", "2 pi"]
        deg_rot = [30, 45, 60, 90, 180, 120, 135, 150, 210, 225, 240, 270, 300, 315, 330, 360]
        rads = [" radians", " rads", ""]
        deg_choice = random.randint(0, len(pi_rot)-1)
        return (pi_rot[deg_choice] + random.choice(rads), deg_rot[deg_choice])

    elif rot_type == "%":
      percentage = random.randint(0, 100)
      deg = (float(percentage) / 100.00) * 360.00
      return (str(percentage) + "%", deg)

    elif rot_type == "fraction":
      frac_rot = ["1/2", "1/4", "1/3", "1/8", "3/4", "half", "quarter", "three quarters", "3 quarters", "a third", "an eighth"]
      deg_rot = [180, 90, 120, 45, 270, 180, 90, 270, 270, 120, 45]
      deg_choice = random.randint(0, len(frac_rot) -1)
      return (str(frac_rot[deg_choice]) + " of a circle", deg_rot[deg_choice])

  # Simple Randomizer
  def rand_inst():
    inst_types = ["FWD1", "FWD2",  "LROT1", "LROT2", "RROT1", "RROT2", "LSIDE1", "LSIDE2", "RSIDE1", "RSIDE2", "BACK1", "BACK2", "DIAGONAL LEFT FORWARD", "DIAGONAL RIGHT FORWARD", "X METERS AT ANGLE Y LEFT", "X METERS AT ANGLE Y RIGHT"]
    rephrase_move = ["move", "go", "advance", "coast", "glide", "get yourself", "move yourself", "proceed"]
    prepositions = ["", "by", "a distance of", "for", "for a total distance of", "equal to", "by a measure of", "about", "by about", "about", "around"]
    prepositions2 = ["", "by", "a distance of", "for", "for a total distance of", "equal to", "by a measure of", "about", "by about", "about", "around"]
    rephrase_forward = ["forward", "onwards", "in front of you", "straight ahead"]
    rephrase_rotate = ["turn", "shift", "rotate", "spin", "turn yourself", "shift yourself", "rotate yourself", "spin"]
    rephrase_lrot = ["to the left", "to your left", "counterclockwise", "CCW", "leftward"]
    rephrase_rrot = ["to your right", "to the right", "clockwise", "CW", "rightward"]
    rephrase_lside = ["to your left", "to the left", "leftwards", "to your relative west"]
    rephrase_rside = ["to your right", "to the right", "rightwards", "to your relative east"]
    rephrase_back1 = ["behind you", "backwards", "rearward", "to your rear", "to the spot behind you"]
    rephrase_back2 = ["behind you", "backwards", "in reverse", "rearward", "to your rear"]

    randtype = random.choice(inst_types)

    if randtype == "FWD1":
      dist =  prompt_randomizer.rand_dist()
      return (random.choice(rephrase_move) + " " +  random.choice(prepositions) +" " +  dist[0] + " " +  random.choice(rephrase_forward), "(MOVE, " + str(round(dist[1],2)) + ")", [("MOVE", round(dist[1],2))])
    elif randtype == "FWD2":
      dist =  prompt_randomizer.rand_dist()
      return (random.choice(rephrase_move) + " " +   random.choice(rephrase_forward) + " " + random.choice(prepositions) +" " +  dist[0], "(MOVE, " + str(round(dist[1],2)) + ")", [("MOVE", round(dist[1],2))])
    elif randtype == "LROT1":
      rot = prompt_randomizer.rand_rot()
      return (random.choice(rephrase_rotate)  + " " +  rot[0] + " " +  random.choice(rephrase_lrot), "(LEFT, " + str(round(rot[1], 2)) + ")", [("LEFT", round(rot[1], 2))])
    elif randtype == "LROT2":
        rot = prompt_randomizer.rand_rot()
        return (random.choice(rephrase_rotate) + " " +  random.choice(rephrase_lrot) + " " + rot[0], "(LEFT, " + str(round(rot[1], 2)) + ")", [("LEFT", round(rot[1], 2))])
    elif randtype == "RROT1":
      rot = prompt_randomizer.rand_rot()
      return (random.choice(rephrase_rotate)  + " " +  rot[0] + " " +  random.choice(rephrase_rrot), "(RGHT, " + str(round(rot[1], 2)) + ")", [("RGHT", round(rot[1], 2))])
    elif randtype == "RROT2":
        rot = prompt_randomizer.rand_rot()
        return (random.choice(rephrase_rotate) + " " +  random.choice(rephrase_rrot) + " " + rot[0], "(RGHT, " + str(round(rot[1], 2)) + ")", [("RGHT", round(rot[1], 2))])
    elif randtype == "LSIDE1":
        dist =  prompt_randomizer.rand_dist()
        return (random.choice(rephrase_move) + " " +  random.choice(prepositions) + " " +  dist[0] + " " +  random.choice(rephrase_lside), "(LEFT, 90), (MOVE, " + str(round(dist[1],2)) +")", [("LEFT", 90), ("MOVE", round(dist[1],2))])
    elif randtype == "LSIDE2":
      dist =  prompt_randomizer.rand_dist()
      return (random.choice(rephrase_move) + " " +  random.choice(rephrase_lside) + " " + random.choice(prepositions2) + " " +  dist[0], "(LEFT, 90), (MOVE, " + str(round(dist[1],2)) +")", [("LEFT", 90), ("MOVE", round(dist[1],2))])
    elif randtype == "RSIDE1":
      dist =  prompt_randomizer.rand_dist()
      return (random.choice(rephrase_move) + " " +  random.choice(prepositions) + " " +  dist[0] + " " +  random.choice(rephrase_rside), "(RGHT, 90), (MOVE, " + str(round(dist[1],2)) +")", [("RGHT", 90), ("MOVE", round(dist[1],2))])
    elif randtype == "RSIDE2":
      dist =  prompt_randomizer.rand_dist()
      return (random.choice(rephrase_move) + " " +  random.choice(rephrase_rside) + " " + random.choice(prepositions2) + " " +  dist[0], "(RGHT, 90), (MOVE, " + str(round(dist[1],2)) +")", [("RGHT", 90), ("MOVE", round(dist[1],2))])
    elif randtype == "BACK1":
      dist =  prompt_randomizer.rand_dist()
      return (random.choice(rephrase_move) + " " +  dist[0] + " " + random.choice(rephrase_back1), "(RGHT, 180), (MOVE, " + str(round(dist[1],2)) +")", [("RGHT", 180), ("MOVE", round(dist[1],2))])
    elif randtype == "BACK2":
      dist =  prompt_randomizer.rand_dist()
      return (random.choice(rephrase_move) + " " + random.choice(rephrase_back2)+ " " +  random.choice(prepositions) + " " + dist[0], "(RGHT, 180), (MOVE, " + str(round(dist[1],2)) +")", [("RGHT", 180), ("MOVE", round(dist[1],2))])
    elif randtype == "DIAGONAL LEFT FORWARD":
      dist =  prompt_randomizer.rand_dist()
      return (random.choice(rephrase_move) + " " + dist[0] + " diagonally to your left", "(LEFT, 45), (MOVE, " + str(round(dist[1],2)) +")", [("LEFT", 45), ("MOVE", round(dist[1],2))])
    elif randtype == "DIAGONAL RIGHT FORWARD":
      dist =  prompt_randomizer.rand_dist()
      return (random.choice(rephrase_move) + " " +  dist[0] + " diagonally to your right", "(RGHT, 45), (MOVE, " + str(round(dist[1],2)) +")", [("RGHT", 45), ("MOVE", round(dist[1],2))])
    elif randtype == "X METERS AT ANGLE Y LEFT":
      dist =  prompt_randomizer.rand_dist()
      rot = prompt_randomizer.rand_rot()
      return (random.choice(rephrase_move) + " " +  dist[0] + " at " + rot[0] + " " + random.choice(rephrase_lrot) , "(LEFT, " + str(round(rot[1], 2)) + "), (MOVE, " + str(round(dist[1],2)) +")", [("LEFT", round(rot[1], 2)), ("MOVE", round(dist[1],2))])
    elif randtype == "X METERS AT ANGLE Y RIGHT":
      dist =  prompt_randomizer.rand_dist()
      rot = prompt_randomizer.rand_rot()
      return (random.choice(rephrase_move) + " " +  dist[0] + " at " + rot[0] + " " + random.choice(rephrase_rrot) , "(RGHT, " + str(round(rot[1], 2)) + "), (MOVE, " + str(round(dist[1],2)) +")", [("RGHT", round(rot[1], 2)), ("MOVE", round(dist[1],2))])
  
  def compute_total(x, init):      
      #print(str(x[0]))  
      #print(str(x[1]))
      if x[0] == "MOVE":
      	init[0] = init[0] + x[1]
      	#init[0] = init[0] + 1
      
      elif x[0]  == "LEFT":
      	init[1] = init[1] - x[1]
      	#init[1] = init[1] - 1
      
      elif x[0]  == "RGHT":
      	init[1] = init[1] + x[1]
      	#init[1] = init[1] + 1
      
      return init
  
  def prompt_maker(max_per_prompt = 3):
    # maximum number of steps per prompt ; does not count in repeated steps (ex. 2x)
    prompt = ""
    no_of_insts = random.randint(1, max_per_prompt)
    
    #no_of_insts = prompt_randomizer.rand_insts(max_per_prompt)
    init = [0.0, 0.0]  # initialize randomizer
    computed_move = []
    #no_of_insts = 5
    simple_move = []
    x = 0
    #print("no of insts: ", no_of_insts)
    
    polite = ('please ', 'show me how you can ', '')
    prompt += random.choice(polite)
    
    #for x in range(no_of_insts):
    while x < no_of_insts:
      #print(x)
      if (no_of_insts) == 2 and (x != 0):
      	add = [' and ', ' then ', '. Please also ', ', and ', ', then ', '. Then, ']
      	to_add = random.choice(add) 
      	
      	if to_add == '. Please also ' or to_add == '. Then, ' or random.randint(0, 10) >= 7:   
            prompt = prompt.capitalize()
            
      	prompt += to_add
        
      elif no_of_insts > 2:
      	if (x < no_of_insts - 1) and (x != 0):
            add = [', ', ', then ']
            prompt += random.choice(add)
      	elif (x == no_of_insts - 1) and (x != 0):
            addition = [', and ', ', then ', ', finally, ']
            prompt += random.choice(addition)
      	elif x == 0:
            prompt = prompt.capitalize()        

      new_inst = prompt_randomizer.rand_inst()
      prompt += new_inst[0]
      simple_move.append(new_inst[1])
      
      for y in new_inst[2]:
      	init = prompt_randomizer.compute_total(y, init)
      	computed_move.append(init[:])

      #max_per_prompt -= 1
      x += 1
      
      # of times
      a = random.randint(0, 10)
      if a > 8:
        multi = prompt_randomizer.rand_multiples(no_of_insts - x)
        prompt += " " + multi[0]
        for b in range(multi[1]):
          simple_move.append(new_inst[1])
          
          for c in new_inst[2]:
          	init = prompt_randomizer.compute_total(c, init)
          	computed_move.append(init[:])
          x += 1
          	
      
    
    #if random.randint(0, 10) >= 7:
    #	prompt = prompt.capitalize()
    	
    if random.randint(1, 2) == 2:
    	prompt += "."
    
    #prompt = prompt.capitalize() 
       
    return (prompt, simple_move, computed_move)
    
#testing
#prompt1 = prompt_randomizer.prompt_maker(5)
#print("Prompt: ", prompt1[0])
#print("Single: ", prompt1[1])
#print("Cumulative: ", prompt1[2])

#print("\n")

#prompt2 = prompt_randomizer.prompt_maker(8)
#print("Prompt: ", prompt2[0])
#print("Single: ", prompt2[1])
#print("Cumulative: ", prompt2[2])

#print("\n")

#prompt3 = prompt_randomizer.prompt_maker(1)
#print("Prompt: ", prompt3[0])
#print("Single: ", prompt3[1])
#print("Cumulative: ", prompt3[2])
  
    