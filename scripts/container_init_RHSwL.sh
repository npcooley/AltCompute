#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=devel_container
#SBATCH --output=devel_container.log

###### -- overhead notes and things -------------------------------------------
# notes and things

# brief description:
# R(ed)H(at)w(ith)L(mod)_container_init.sh
# using slurm to spin up a container on Red Hat with Lmod modules
# this script was designed around the Culhane Lab's compute node setup and 
# containers based on Rocker's RStudio container, this script prints out things
# to the log file, like which compute node the job landed on, and which port is open
# a typical invocation would like:
# sbatch <this_script.sh> docker://npcooley/turkuhackathon:0.0.4 cuda11.8/toolkit/11.8.0 apptainer/1.1.9
# however
# sbatch <this_script.sh> docker://npcooley/turkuhackathon:0.0.4
# should also work and will find the most up-to-date version of apptainer and the cuda toolkit

# some notes:
# don't hard code ports
# nuke tmp files and directories on exit
# hypothetically, when the job is killed by slurm, the port will be released
# notes cannot be placed above the sbatch block!
# the container needs to inherit the parts of the host environment that are
# consequential to the cuda installation, this includes the host compiler

# dependencies / requirements:
# Lmod ................ https://lmod.readthedocs.io/en/latest/010_user.html
# CUDA ................ https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
# apptainer ........... https://apptainer.org/docs/user/latest/
# slurm ............... https://slurm.schedmd.com/sbatch.html
# rstudio container ... https://rocker-project.org/

# partially based on:
# Roberts Lab RStudio Server guide:
# https://robertslab.github.io/resources/klone_RStudio-Server/

# arguments and overhead stuff...
# `module avail` sends it's output to the stderr, so we need to redirect
# it to capture
# we only want to accept three different positional arguments here
# 1 == container name -- errors if not supplied, designed around docker://npcooley/turkuhackathon:0.0.4
# 2 == cuda module -- default to most recent version from avail resp
# 3 == apptainer module -- default to most recent version from avail resp

###### -- clean up and trap ---------------------------------------------------

# cleanup function ... if we exit early, clean up after ourselves
cleanup() {
  echo "cleaning up temporary files..."
  rm -rf ${RSTUDIO_TMP}
  rm -f ${mod_avl_tmp}
  echo "done"
}
trap cleanup EXIT SIGTERM SIGINT

###### -- user args -----------------------------------------------------------

container_name=${1:-""}
cuda_module=${2:-""}
apptainer_module=${3:-""}
mod_avl_tmp=$(mktemp)
# replace white spaces with a new line with tr
# then find every line that matches the pattern start of line : end of line
# and inver the pattern with `-v`
module avail 2>&1 | tr ' ' '\n' | grep -v '^$' > ${mod_avl_tmp}

# error if container name is not supplied
# `-z` is technically only testing whether the string is of length zero, but
# we'll live with that for now
if [ -z "${container_name}" ]; then
  echo "======"
  echo "container name is *required* as the first argument"
  echo "======"
  echo " "
  exit 1
fi

# loading modules directly ...
# module load cuda11.8/toolkit/11.8.0
# module load apptainer/1.1.9

###### -- other functions for this script -------------------------------------

# accept a single keyword, an optional user supplied pattern, and a file with one module per line
# for cuda this is just `cuda[^/]*/toolkit/[0-9]+\.[0-9]+\.[0-9]+` as the pattern
# for apptainer this is just `apptainer/[0-9]+\.[0-9]+\.[0-9]+`
# if the user supplies a pattern, just ensure that it exists in the captured response
load_target_module() {
  local curr_modules=$1
  local default_pattern=$2
  local user_pattern=${3:-""}
  local pattern_hit
  
  # if a user supplied pattern is present, make sure it is available to be loaded
  if [ -n "${user_pattern}" ]; then
    if grep -qF "${user_pattern}" "${curr_modules}"; then
      echo "  ======"
      echo "  ${user_pattern} was found!"
      echo "  ======"
      echo " "
      module load ${user_pattern}
      return 0      
    else
      # the supplied pattern was not found
      echo "  ======"
      echo "  ${user_pattern} was not found in the list of available modules, exiting..."
      echo "  ======"
      echo " "
      return 1
    fi
  else
    # if a user supplied pattern is not present, find all instances of the default keyword
    # and select the most recent by escalating version number
    # -E is extended regular expressions
    # -o returns the captured characters only ... will that be a problem?
    pattern_hit=$(grep -oE "${default_pattern}" "${curr_modules}" | sort -V | tail -1)
    if [ -n "${pattern_hit}" ]; then
      echo "  ======"
      echo "  ${pattern_hit} was found matching a default pattern and will be loaded!"
      echo "  ======"
      module load ${pattern_hit}
      return 0
    else
      echo "  ======"
      echo "  the pattern '${default_pattern}' failed to return any hits, please check 'module avail'!"
      echo "  ======"
      echo " "
      return 1
    fi
  fi
  
}

# find an available non-restricted port:
find_free_port() {
  # local variables for a bash function
  local port
  local port_found
  # loop through 100 ports from the traditionally user acceptable ranges
  for port in $(shuf -i 1024-65535 -n 100); do
    # is the port already in use
    ss -tlnp | grep -q ":${port} "
    port_found=$?
    # if the randomly selected port is not found on the list
    if [ ${port_found} -ne 0 ]; then
      echo ${port}
      return 0
    fi
  done
  # if i get here without reaching the return statement in the loop
  # none of the randomly selected ports were available, return 1
  return 1
}

###### -- module loading ------------------------------------------------------

load_target_module ${mod_avl_tmp} "cuda[^/]*/toolkit/[0-9]+\.[0-9]+\.[0-9]+" "${cuda_module}"
if [ $? -ne 0 ]; then
  echo "======"
  echo "failed to load an appropriate CUDA module"
  echo "======"
  echo " "
  exit 1
fi
load_target_module ${mod_avl_tmp} "apptainer/[0-9]+\.[0-9]+\.[0-9]+" "${apptainer_module}"
if [ $? -ne 0 ]; then
  echo "======"
  echo "failed to load an appropriate apptainer module"
  echo "======"
  echo " "
  exit 1
fi

###### -- directories and apptainer args --------------------------------------

# CUDA bindings that need to be explicit:
# translate colon to new line
# find cuda case insensitive
# translate new lines to colons
# drop last colon
avl_cuda_libs=$(echo $LD_LIBRARY_PATH | \
  tr ':' '\n' | \
  grep -i 'cuda' | \
  tr '\n' ':' | \
  sed 's/:$//')
  
# there should be two cuda lib locations, a driver side, and a toolkit side
# check for them, error if the collection attempt is empty
if [ -z "${avl_cuda_libs}" ]; then
  echo "======"
  echo "no CUDA entries found in LD_LIBRARY_PATH — please check module loading..."
  echo "======"
  exit 1
else
  echo "======"
  echo "CUDA library paths:"
  echo "${avl_cuda_libs}" | tr ':' '\n'
  echo "======"
fi

# apptainer *should* append this to the container's LD_LIBRARY_PATH
export APPTAINERENV_LD_LIBRARY_PATH=${avl_cuda_libs}

# this variable may not be supported on all versions of the apptainer module?
export APPTAINERENV_APPEND_PATH="${CUDA_HOME}/bin"

host_gcc_libs=$(echo $LD_LIBRARY_PATH | \
  tr ':' '\n' | \
  grep -i 'gcc' | \
  tr '\n' ':' | \
  sed 's/:$//')

if [ -z "${host_gcc_libs}" ]; then
  echo "======"
  echo "no GCC entries found in LD_LIBRARY_PATH — please check module loading..."
  echo "======"
  exit 1
else
  echo "======"
  echo "GCC library paths:"
  echo "${host_gcc_libs}" | tr ':' '\n'
  echo "======"
fi

host_gcc_bin=$(dirname $(which gcc))

if [ -z "${host_gcc_bin}" ]; then
  echo "======"
  echo "gcc bin appears absent, please echo the host's PATH"
  echo "======"
  exit 1
else
  echo "======"
  echo "gcc found: $(${host_gcc_bin}/gcc | head -n 1)"
  echo "======"
fi

host_gcc_home=$(dirname ${host_gcc_bin})


# working directory - where your SIF will be cached and R libraries stored
RSTUDIO_CWD=$HOME

# create temp directories for ephemeral runtime content
RSTUDIO_TMP=$(mktemp -d)
echo "======"
echo "rserver logs present at: ${RSTUDIO_TMP}"
echo "to examine them, remove this script's 'trap' statement"
echo "======"

mkdir -p -m 700 \
  ${RSTUDIO_TMP}/run \
  ${RSTUDIO_TMP}/run/lock/rstudio-server \
  ${RSTUDIO_TMP}/tmp \
  ${RSTUDIO_TMP}/var/lib/rstudio-server \
  ${RSTUDIO_TMP}/var/log/rstudio/rstudio-server \
  ${RSTUDIO_TMP}/tmp/R_libs


# database config
cat > ${RSTUDIO_TMP}/database.conf <<capture_this
provider=sqlite
directory=/var/lib/rstudio-server
capture_this

cat > ${RSTUDIO_TMP}/logging.conf <<capture_this
[*]
log-level=warn
logger-type=file
log-dir=/var/log/rstudio
capture_this

# rsession wrapper script
# if we quote our heredoc, i.e. $ whatever <<'capture_this'
# ...
# capture_this
# we do not perform shell expansion during capture
cat > ${RSTUDIO_TMP}/rsession.sh <<capture_this
#!/bin/sh
export OMP_NUM_THREADS=${SLURM_JOB_CPUS_PER_NODE}
export R_LIBS_USER=/tmp/R_libs
export PATH="${host_gcc_home}/bin:${CUDA_HOME}/bin:\${PATH}"
export LD_LIBRARY_PATH="${host_gcc_libs}:${avl_cuda_libs}:\${LD_LIBRARY_PATH}"
exec /usr/lib/rstudio-server/bin/rsession "\${@}"
capture_this

chmod +x ${RSTUDIO_TMP}/rsession.sh

# bind mounts
export APPTAINER_BIND="\
${RSTUDIO_TMP}/run:/run,\
${RSTUDIO_TMP}/tmp:/tmp,\
${RSTUDIO_TMP}/var/lib/rstudio-server:/var/lib/rstudio-server,\
${RSTUDIO_TMP}/var/log/rstudio:/var/log/rstudio,\
${RSTUDIO_TMP}/database.conf:/etc/rstudio/database.conf,\
${RSTUDIO_TMP}/rsession.sh:/etc/rstudio/rsession.sh,\
${RSTUDIO_TMP}/logging.conf:/etc/rstudio/logging.conf,\
${host_gcc_home}:${host_gcc_home},\
${CUDA_HOME}:${CUDA_HOME}"

# environment variables inside container
export APPTAINERENV_RSTUDIO_SESSION_TIMEOUT=0
export APPTAINERENV_USER=$(id -un)
export APPTAINERENV_PASSWORD=$(openssl rand -base64 15)

# test call to rserver
# RSERVER_PATH=$(apptainer exec --cleanenv \
#   docker://npcooley/turkuhackathon:0.0.4 \
#   which rserver)
# EXIT_CODE=$?
# echo "rserver path: ${RSERVER_PATH}"
# echo "which rserver exited with: ${EXIT_CODE}"

###### -- find an available port ----------------------------------------------

# assume failure until proven otherwise...
check_count=1
port_check=1
while [ ${check_count} -le 3 ] && [ ${port_check} -ne 0 ]; do
  echo "looking for an available port, try number ${check_count}..."
  echo " "
  avl_port=$(find_free_port)
  port_check=$?
  # double brackets here 
  ((check_count++))
done

if [ ${port_check} -eq 1 ]; then
  echo "======"
  echo "no available port was found after a reasonable number of tries..."
  echo "please check port usage"
  echo "======"
  exit 1
else
  echo "======"
  echo "available port found at: ${avl_port}"
  echo "======"
fi

###### -- prefs file checks ---------------------------------------------------

# because we're setting a user, prefs files like the:
# /home/rstudio/.config/rstudio/rstudio-prefs.json
# file provided in a container won't be respected, because the user will not be
# `rstudio`
# we can get around this by checking for the existence of prefs files both locally
# and in selected container

# this logic block is silent if no prefs file is found locally, or on the container...
if [ -s "${HOME}/.config/rstudio/rstudio-prefs.json" ]; then
  # create the config directory structure in tmp
  echo "======"
  echo "user defined rstudio prefs exist on the host environment and will be used..."
  echo "======"
  
  # append to the existing APPTAINER_BIND
  export APPTAINER_BIND="${APPTAINER_BIND},\
${HOME}/.config/rstudio/rstudio-prefs.json:${HOME}/.config/rstudio/rstudio-prefs.json"
else
  # run a test call to the container...
  apptainer exec --cleanenv \
    ${container_name} \
    cat /home/rstudio/.config/rstudio/rstudio-prefs.json \
    > ${RSTUDIO_TMP}/rstudio-prefs.json
    
  if [ $? -eq 0 ] && [ -s "${RSTUDIO_TMP}/rstudio-prefs.json" ]; then
    echo "======"
    echo "user defined rstudio prefs exist in the container and will be used..."
    echo "======"
    # create the config directory structure in tmp
    mkdir -p -m 700 ${RSTUDIO_TMP}/rstudio_config
    mv ${RSTUDIO_TMP}/rstudio-prefs.json ${RSTUDIO_TMP}/rstudio_config/rstudio-prefs.json
    
    # `-p` should do nothing if the directory already exists ...
    mkdir -p -m 700 ${HOME}/.config/rstudio
    
    # bind mounts into tmp are cleaned up after exit, and should not be inherited
    # back to the host
    export APPTAINER_BIND="${APPTAINER_BIND},\
${RSTUDIO_TMP}/rstudio_config/rstudio-prefs.json:${HOME}/.config/rstudio/rstudio-prefs.json"
  fi
  # nothing to do here
fi

###### -- call apptainer ------------------------------------------------------

echo "RStudio credentials:"
echo "  user: ${APPTAINERENV_USER}"
echo "  password: ${APPTAINERENV_PASSWORD}"
echo "  tunnel: ssh -N -L 8787:${HOSTNAME}:${avl_port} ${APPTAINERENV_USER}@<loginnode>"
echo "  note: '<loginnode>' is the resource you ssh'd into initially"

# if you want to avoid logging in every time, `auth-stay-signed-in-days=XX`
# can be set to some no-zero number, this script was originally generated with
# 30 here, but has been changed to zero for the sake of my sanity
apptainer exec --cleanenv \
  --nv \
  --home ${RSTUDIO_CWD} \
  ${container_name} \
  rserver \
    --www-port=${avl_port} \
    --auth-none=0 \
    --auth-pam-helper-path=pam-helper \
    --auth-stay-signed-in-days=0 \
    --auth-timeout-minutes=0 \
    --rsession-path=/etc/rstudio/rsession.sh \
    --server-user=${APPTAINERENV_USER}


