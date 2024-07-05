test_acapulco:	test_acapulco.cpp
	g++ -o test_acapulco test_acapulco.cpp -std=c++11 -O2 -I/opt/OpenBLAS/include/ -DARMA_DONT_USE_WRAPPER  -L/opt/OpenBLAS/lib -lopenblas
