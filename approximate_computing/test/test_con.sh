nohup python -u test_con.py --model_path '../train_atheta/train_files/con/train_a_con_51lr0001/' --gpu 7 --proxy_path '../train_proxy/train_files/con/train_con_100lr0001/' --testset_path '../build_dataset/testset/' --save_path './test_results/test_con/' > ./test_con_100lr0001_30.log  2>&1 </dev/null &