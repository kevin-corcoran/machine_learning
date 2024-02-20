	ass_per = (60.25+89.41+100+97.4)/(72+108+200)
	mid = 46.64/50
	# mid = 0.94
	part = 0.61
	fin = 0.9

  function final_score(ass, mid, part, final)
    return ass*0.4 + mid*0.2 + part*0.05 + final*0.35
  end

  println(final_score(ass_per, mid, part, fin))

