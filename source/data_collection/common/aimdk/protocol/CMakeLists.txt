cmake_minimum_required(VERSION 3.16)

# -------------------------------------------------------------
# Create the project

aimrte_project(NAME aimrt_comm_protocol ROOT_NAMESPACE "aimdk;protocol")
aimrte_add_subdirectories(.)
aimrte_project_complete()
