#!/bin/bash

# brief description:
# container_init_brev_generic.sh
# spin up a Rocker RStudio Server container on an NVIDIA Brev instance using Docker
# prints credentials and the SSH tunnel helpers needed to reach the server
#
# explicit invocation, where <my_cuda_version> is cuda-12.2 or whatever:
# <this_script_name> npcooley/turkuhackathon:0.0.4 /usr/local/<my_cuda_version>
# invocation where we just auto-detect the most recent cuda version:
# bash <this_script_name> npcooley/turkuhackathon:0.0.4
#
# dependencies / requirements:
# docker ............. https://docs.docker.com/engine/install/
# brev CLI ........... https://docs.brev.dev/
#
# GPU inheritance strategy (informed by prior testing on Brev):
#   --gpus all          passes NVIDIA devices and injects driver-matched libraries
#                       do NOT bake nvidia-opencl-dev into the image; it will
#                       produce a driver/library version mismatch at runtime
#   -v /etc/OpenCL/vendors/:/etc/OpenCL/vendors/
#                       inherits the host's OpenCL ICD files so the ICD loader
#                       inside the container can find the NVIDIA platform
#   -v <cuda_path>:/usr/local/cuda:ro  (+ PATH / LD_LIBRARY_PATH)
#                       inherits nvcc and the toolkit from the host; keeps the
#                       image lightweight and avoids pinning a specific CUDA
#                       version in the Dockerfile
#
# notes:
# don't hard code ports
# container is removed on exit (--rm is set in the docker run call)
# the generated password is ephemeral; it is not written to disk

###### -- args and validation -------------------------------------------------

# ${parameter:-default}
# use $parameter if it is present, if it is not, use default
container_name="${1:-""}"
cuda_path_override="${2:-""}"
ocl_path_override="${3:-""}"

if [ -z "${container_name}" ]; then
  echo "======"
  echo "error: container name is required as the first argument"
  echo "usage: bash $(basename "${0}") <container_name> [optional_cuda_path]"
  echo "======"
  exit 1
fi

###### -- ad hoc functions ----------------------------------------------------

# command availability checking
check_host_dependency() {
  local tool_name="${1}"
  if ! command -v "${tool_name}" > /dev/null 2>&1; then
    echo "======"
    echo "error: '${tool_name}' was not found on PATH; is it installed?"
    echo "======"
    return 1
  fi
  return 0
}

# generic tool check
# prefer an explicit override, otherwise probe the canonical symlink,
# then fall back to finding the highest versioned directory under /usr/local
# invocation:
# first two variables set in this script, last can be set by user supplied
# var *to* this script
# pathfinder_best_attempt <my_path> <assumed altform> <user_supplied_path>
# i.e.
# pathfinder_best_attempt /usr/local/cuda cuda-* "${my_var_that_can_be_empty}"
pathfinder_best_attempt() {
  local assumed_canon="$1"
  local alternative_pattern="$2"
  local user_override="$3"
  local pattern_found
  
  if [ -n "${user_override}" ]; then
    if [ -d "${user_override}" ]; then
      echo "${user_override}"
      return 0
    else
      echo "warning: a user override was supplied but does not exist" >&2
    fi
  fi
  
  if [ -d "${assumed_canon}" ]; then
    echo "${assumed_canon}"
    return 0
  fi
  
  pattern_found=$(find /usr/local \
    -maxdepth 1 \
    -type d \
    -name "${alternative_pattern}" \
    2>/dev/null \
    | sort -V \
    | tail -1)
  
  if [ -n "${pattern_found}" ]; then
    echo "${pattern_found}"
    return 0
  fi
  
  # if we got here, the directory was not found
  return 1
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


###### -- overhead checks and host interrogation ------------------------------

check_host_dependency "docker"   || exit 1
check_host_dependency "ss"       || exit 1
check_host_dependency "openssl"  || exit 1


cuda_host_path=$(pathfinder_best_attempt /usr/local/cuda cuda-* "${cuda_path_override}")
cuda_found=$?

if [ "${cuda_found}" -ne 0 ]; then
  echo "======"
  echo "no CUDA toolkit was discovered, please check the host environment"
  echo "======"
  exit 1
else
  echo "======"
  echo "CUDA toolkit implied to be at ${cuda_host_path} on the host"
  echo "======"
  # bind the toolkit from the resource instance to the container instance
  cuda_toolkit_bind_mount="-v ${cuda_host_path}:/usr/local/cuda:ro"
  # make changes to the container instance's environment variables
  cuda_env_path="-e PATH=/usr/local/cuda/bin:\${PATH}"
  cuda_env_ldpath="-e LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}"
fi

ocl_host_path=$(pathfinder_best_attempt /etc/OpenCL/vendors OpenCL "${ocl_path_override}")
ocl_found=$?

if [ "${ocl_found}" -ne 0 ]; then
  echo "======"
  echo "OpenCL vendor info not found, OpenCL will not be operable in the container"
  echo "======"
  ocl_bind_mount=""
else
  echo "======"
  echo "OpenCL .icd files found"
  echo "======"
  ocl_bind_mount="-v ${ocl_host_path}:/etc/OpenCL/vendors:ro"
fi

# get a port
echo "======"
echo "looking for an available port..."
check_count=1
port_check=1
avl_port=""

while [ ${check_count} -le 3 ] && [ ${port_check} -ne 0 ]; do
  avl_port=$(find_free_port)
  port_check=$?
  ((check_count++))
done

if [ ${port_check} -ne 0 ]; then
  echo "  error: no available port found after a reasonable number of tries"
  echo "  please check port usage with: ss -tlnp"
  echo "======"
  exit 1
fi

echo "  available port found at: ${avl_port}"
echo "======"

###### -- credential generation -----------------------------------------------

rstudio_user=$(id -un)
rstudio_password=$(openssl rand -base64 15)
container_instance_name="rstudio_$(date +%s)"

###### -- docker call construction and execution ------------------------------

# R libraries bind mount: persist installed packages across container restarts
# by storing them in a host-side directory under $HOME
r_libs_host="${HOME}/R/docker"
mkdir -p "${r_libs_host}"

echo "======"
echo "starting RStudio container..."
echo "======"

# NOTE: --rm means the container is removed when it exits; R libraries
# installed inside the session are only persisted if R_LIBS_USER
# points to the bind-mounted host directory above
docker run -d \
  --name "${container_instance_name}" \
  --gpus all \
  -p "${avl_port}":8787 \
  -e PASSWORD="${rstudio_password}" \
  -e "DISABLE_AUTH=false" \
  -e "R_LIBS_USER=/home/rstudio/R/docker" \
  -v "${r_libs_host}":/home/rstudio/R/docker \
  ${ocl_icd_bind_mount} \
  ${cuda_bind_mount} \
  ${cuda_env_path} \
  ${cuda_env_ldpath} \
  "${container_name}" \

# give the server a moment to start before printing credentials,
# so the log output doesn't interleave with the credential block
sleep 3

echo " "
echo "======"
echo "RStudio credentials:"
echo "  user:     ${rstudio_user}"
echo "  password: ${rstudio_password}"
echo "  tunnel:   brev port-forward <instance-name> --port 8787:${avl_port}"
echo "  note:     replace <instance-name> with the name you used in 'brev create'"
echo "  note:     open http://localhost:8787 in your browser after the tunnel is established"
echo "======"
echo "container name: ${container_instance_name}"
echo "to stop the server: docker stop ${container_instance_name}"
echo "======"
