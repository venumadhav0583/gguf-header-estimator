set -x
cd "$(dirname "$0")"; export PATH=$HOME/.local/bin:$PATH
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp > clone.log 2>&1
cd llama.cpp && git log -1 --format='%h %cd' > ../llama_commit.txt
cmake -B build -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF > ../cmake.log 2>&1
cmake --build build --config Release -j 2 --target llama-cli llama-completion llama-simple llama-bench llama-tokenize > ../build.log 2>&1
echo BUILD_DONE $? >> ../status.txt
