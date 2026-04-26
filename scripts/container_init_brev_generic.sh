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

if [ -z "${container_name}" ]; then
  echo "======"
  echo "error: container name is required as the first argument"
  echo "usage: bash $(basename "${0}") <container_name> [cuda_path]"
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

# cuda check (explicit to cuda, not generalized)
# prefer an explicit override, otherwise probe the canonical symlink,
# then fall back to finding the highest versioned directory under /usr/local
find_cuda_path() {
  local override="${1}"

  if [ -n "${override}" ]; then
    if [ -d "${override}" ]; then
      echo "${override}"
      return 0
    else
      echo "  ======"
      echo "  warning: supplied cuda path '${override}' does not exist, falling back to auto-detection"
      echo "  ======"
    fi
  fi

  # canonical symlink is the tidiest case
  if [ -d "/usr/local/cuda" ]; then
    echo "/usr/local/cuda"
    return 0
  fi

  # no symlink; find the most recent versioned directory
  local best_path
  best_path=$(find /usr/local -maxdepth 1 -type d -name "cuda-*" 2>/dev/null \
    | sort -V | tail -1)

  if [ -n "${best_path}" ]; then
    echo "${best_path}"
    return 0
  fi

  # nothing found
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


###### -- overhead checks -----------------------------------------------------

check_host_dependency "docker"   || exit 1
check_host_dependency "ss"       || exit 1
check_host_dependency "openssl"  || exit 1

echo "======"
echo "locating CUDA toolkit on host..."
cuda_host_path=$(find_cuda_path "${cuda_path_override}")
cuda_found=$?

if [ ${cuda_found} -ne 0 ]; then
  echo "  warning: no CUDA toolkit directory found on this host"
  echo "  nvcc will not be available inside the container"
  echo "  if this is unexpected, re-run with an explicit path as the second argument"
  echo "  continuing without CUDA bind mount..."
  echo "======"
  cuda_bind_mount=""
  cuda_env_path=""
  cuda_env_ldpath=""
else
  echo "  found CUDA toolkit at: ${cuda_host_path}"
  echo "======"
  cuda_bind_mount="-v ${cuda_host_path}:/usr/local/cuda:ro"
  cuda_env_path="-e PATH=/usr/local/cuda/bin:\${PATH}"
  cuda_env_ldpath="-e LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}"
fi

# opencl's .icd files ...
echo "======"
echo "checking for OpenCL ICD files on host..."
ocl_icd_host_path="/etc/OpenCL/vendors"

if [ -d "${ocl_icd_host_path}" ]; then
  echo "  found OpenCL ICD directory at: ${ocl_icd_host_path}"
  echo "======"
  ocl_icd_bind_mount="-v ${ocl_icd_host_path}:${ocl_icd_host_path}"
else
  echo "  warning: ${ocl_icd_host_path} not found on this host"
  echo "  OpenCL platform discovery may fail inside the container"
  echo "  continuing without OpenCL ICD bind mount..."
  echo "======"
  ocl_icd_bind_mount=""
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
