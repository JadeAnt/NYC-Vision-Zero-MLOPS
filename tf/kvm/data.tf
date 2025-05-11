# v2 refers to the Neutron networking API version
# data "openstack_networking_<resource>_v2" "name_you_pick" {
#  name = "actual-name-in-openstack"
# }

data "openstack_networking_network_v2" "sharednet2" {
  provider = openstack.kvm
  name = "sharednet2"
}

data "openstack_networking_network_v2" "sharednet1" {
  provider = openstack.kvm
  name = "sharednet1"
}

data "openstack_networking_subnet_v2" "sharednet1_subnet" {
  provider = openstack.kvm
  name = "sharednet1-subnet"
}

data "openstack_networking_secgroup_v2" "allow_ssh" {
  provider = openstack.kvm
  name = "allow-ssh"
}

data "openstack_networking_secgroup_v2" "allow_9001" {
  provider = openstack.kvm
  name = "allow-9001"
}

data "openstack_networking_secgroup_v2" "allow_8000" {
  provider = openstack.kvm
  name = "allow-8000"
}

data "openstack_networking_secgroup_v2" "allow_8080" {
  provider = openstack.kvm
  name = "allow-8080"
}

data "openstack_networking_secgroup_v2" "allow_8081" {
  provider = openstack.kvm
  name = "allow-8081"
}

data "openstack_networking_secgroup_v2" "allow_http_80" {
  provider = openstack.kvm
  name = "allow-http-80"
}

data "openstack_networking_secgroup_v2" "allow_9090" {
  provider = openstack.kvm
  name = "allow-9090"
}

data "openstack_networking_secgroup_v2" "allow_3000" {
  provider = openstack.kvm
  name = "allow-3000"
}

data "openstack_networking_secgroup_v2" "allow_8265" {
  provider = openstack.kvm
  name = "allow-8265"
}

# --- CHI@Edge Section ---

/*
data "openstack_networking_network_v2" "edge_sharednet2" {
  provider = openstack.chiedge
  name     = "sharednet2"
}
*/

data "openstack_networking_network_v2" "edge_public" {
  provider = openstack.chiedge
  name     = "public"
}

data "openstack_networking_network_v2" "edge_caliconet" {
  provider = openstack.chiedge
  name     = "caliconet"
}

data "openstack_networking_secgroup_v2" "edge_allow_ssh" {
  provider = openstack.chiedge
  name     = "allow-ssh"
}

data "openstack_networking_secgroup_v2" "edge_allow_8000" {
  provider = openstack.chiedge
  name     = "allow-8000"
}
