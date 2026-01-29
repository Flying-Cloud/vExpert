# Test multiple balanceness values
python tests/test_low_latency.py --balanceness 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256

# Or test with pressure test
# python tests/test_low_latency.py --balanceness 1.0,0.5 --num-processes 8 --pressure-test  