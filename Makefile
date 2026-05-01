clean-runs:
	rm -rf runs/*

clean-recordings:
	rm -rf recordings/*

clean:
	make clean-runs
	make clean-recordings