1.Requirement pakage:
    (1).numpy
    (2).torch
    (3).matplotlib

2.execute with the VScode
3.There are four .py file in the zip : (1)dl_hw2_AE.py (2)dl_hw2_VAE.py (3)hw2_AE_VAE.py (4)models.py
    , they need to be in the same folder.
4.if you want to get "AE" result, run "dl_hw2_AE.py"
    else if you want to get "VAE" result, run "dl_hw2_VAE.py"
    else if you want to get the comparison betwenn "AE" & "VAE", run "hw2_AE_VAE.py"
5."models.py" include AE, VAE model constructure.
6.after running the "dl_hw2_AE.py" & "dl_hw2_VAE.py", it will produce 4 reconstructed pictures(idx = 3, 227, 841, 1475)
7.after running the "hw2_AE_VAE.py", 4 triple pairs of reconstructed pictures that compare between Ori, AE, VAE(idx = 1, 2, 3, 4)