# synthesized-insight

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/synthesized-io/insight/master.svg)](https://results.pre-commit.ci/latest/github/synthesized-io/insight/master)

Installation
--------------

### 1. Install python requirements:
#### macOS
(make sure you have brew installed)
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
```
brew install openssl readline sqlite3 xz zlib
```
#### ubuntu
```
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

### 2. Install pyenv:
#### macOS
```
brew update
brew install pyenv
```
#### ubuntu
```
curl https://pyenv.run | bash
```
### 3. Install python 3.8.10:
```
pyenv install 3.8.10
```
