python3 fit_test.py --Nb 20 --loc_path results/ --tem_path BHfit_test/ --width 1 --bkg uni --Nbkg 100000 > ~/scratch/out.txt 2> err.txt &
python3 fit_test.py --Nb 20 --loc_path results/ --tem_path BHfit_test/ --width 5 --bkg uni --Nbkg 100000 > ~/scratch/out.txt 2> err.txt &
python3 fit_test.py --Nb 20 --loc_path results/ --tem_path BHfit_test/ --width 1 --bkg exp --Nbkg 1000 > ~/scratch/out.txt 2> err.txt &
