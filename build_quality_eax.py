"""Build the upstream Large GA-EAX variant with progress and seed support."""
import argparse
import subprocess
import tempfile
from pathlib import Path


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Unexpected upstream source around {old!r}")
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Upstream GA_EAX_1.0/Large directory")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="eax-build-") as scratch:
        directory = Path(scratch)
        for path in sorted(args.source.iterdir()):
            if path.suffix not in (".cpp", ".h"):
                continue
            source = path.read_text()
            if path.name == "main.cpp":
                source = replace(source, "InitURandom();", """const char* seed = getenv("PROJECT2_EAX_SEED");
  if (seed) InitURandom(atoi(seed)); else InitURandom();""")
            if path.name == "env.cpp":
                source = "#include <cstdlib>\n#include <climits>\n" + source
                source = replace(source, "  while( 1 )\n {\n    this->SetAverageBest();",
                                 """  int writtenBest = INT_MAX;
  while( 1 )
  {
    this->SetAverageBest();
    if (getenv("PROJECT2_EAX_PROGRESS") && fBestValue < writtenBest) {
      char filename[] = "progress";
      this->WriteBest(filename);
      writtenBest = fBestValue;
    }""")
            (directory / path.name).write_text(source)
        compiled = directory / "eax"
        subprocess.run(["g++", "-O3", *map(str, sorted(directory.glob("*.cpp"))), "-o", str(compiled)],
                       check=True)
        compiled.replace(output)


if __name__ == "__main__":
    main()
