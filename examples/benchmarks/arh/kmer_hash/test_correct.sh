cat test*.dat | sort > my_solution.txt
diff my_solution.txt /global/project/projectdirs/mp309/cs267-spr2018/hw3-datasets/$1_solution.txt
