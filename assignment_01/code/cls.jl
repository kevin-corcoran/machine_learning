# (TP,TN,FP) P̂ = TP+FP
cls = [(16,0,9), (16,4,5), (13,6,3), (10,7,2), (4,9,0), (0,9,0)]
P = 16; N = 9
i = 0
for c in cls
  global i+=1
  P̂ = cls[i][1] + cls[i][3]
  println("classifier: $i")
  print("accuracy: ")
  println((cls[i][1] + cls[i][2])/(P+N))
  print("precision: ")
  println(cls[i][1]/P̂)
  print("recall: ")
  println(cls[i][1]/P)
  print("\n")
end
