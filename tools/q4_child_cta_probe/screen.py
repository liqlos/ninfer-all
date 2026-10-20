import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ffn_pair_probe'))
import run
run.OUT_SUBDIR = 'q4-child-cta'
run.PROBE_NAME = 'q4-child-cta-probe'
run.main()
