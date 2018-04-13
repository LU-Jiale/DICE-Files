box = load('assignment_1_test_v2.mat');
test_set_v2 = box.test_set_v2;
for i = 1:25
	  imag2d(test_set_v2{i}.Color);
end

