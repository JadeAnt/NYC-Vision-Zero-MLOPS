resource "openstack_networking_network_v2" "private_net" {
  provider = openstack.kvm
  name                  = "private-net-mlops-${var.suffix}"
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet" {
  provider = openstack.kvm
  name       = "private-subnet-mlops-${var.suffix}"
  network_id = openstack_networking_network_v2.private_net.id
  cidr       = "192.168.1.0/24"
  no_gateway = true
}

resource "openstack_networking_port_v2" "private_net_ports" {
  provider = openstack.kvm
  for_each              = var.nodes
  name                  = "port-${each.key}-mlops-${var.suffix}"
  network_id            = openstack_networking_network_v2.private_net.id
  port_security_enabled = false

  fixed_ip {
    subnet_id  = openstack_networking_subnet_v2.private_subnet.id
    ip_address = each.value
  }
}

resource "openstack_networking_port_v2" "sharednet2_ports" {
  provider = openstack.kvm
  for_each   = var.nodes
    name       = "sharednet2-${each.key}-mlops-${var.suffix}"
    network_id = data.openstack_networking_network_v2.sharednet2.id
    security_group_ids = [
      data.openstack_networking_secgroup_v2.allow_ssh.id,
      data.openstack_networking_secgroup_v2.allow_9001.id,                 # minio
      data.openstack_networking_secgroup_v2.allow_8000.id,                 # mlflow
      data.openstack_networking_secgroup_v2.allow_8080.id,                 # canary
      data.openstack_networking_secgroup_v2.allow_8081.id,                 # staging
      data.openstack_networking_secgroup_v2.allow_http_80.id,
      data.openstack_networking_secgroup_v2.allow_9090.id,                  # argo cd, prometheus
      data.openstack_networking_secgroup_v2.allow_3000.id,                 # grafana
      data.openstack_networking_secgroup_v2.allow_8265.id                  # Ray
    ]
}

resource "openstack_compute_instance_v2" "nodes" {
  provider = openstack.kvm
  for_each = var.nodes

  name        = "${each.key}-mlops-${var.suffix}"
  image_name  = "CC-Ubuntu24.04"
  flavor_name = "m1.medium"
  key_pair    = var.key

  network {
    port = openstack_networking_port_v2.sharednet2_ports[each.key].id
  }

  network {
    port = openstack_networking_port_v2.private_net_ports[each.key].id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 ${each.key}-mlops-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF

}

resource "openstack_networking_floatingip_v2" "floating_ip" {
  provider = openstack.kvm
  pool        = "public"
  description = "MLOps IP for ${var.suffix}"
  port_id     = openstack_networking_port_v2.sharednet2_ports["node1"].id
}




# --- Chi@Edge Section ---

resource "openstack_networking_port_v2" "edge_public_port" {
  provider   = openstack.chiedge
  name       = "chi-edge-port-${var.suffix}"
  network_id = data.openstack_networking_network_v2.edge_public.id
  security_group_ids = [
    data.openstack_networking_secgroup_v2.edge_allow_ssh.id,
    data.openstack_networking_secgroup_v2.edge_allow_8000.id        # FastAPI
  ]
}

resource "openstack_compute_instance_v2" "chi_edge_node" {
  provider    = openstack.chiedge
  name        = "chi-edge-${var.suffix}"
  image_name  = "CC-Ubuntu24.04"                  # May need to change
  flavor_name = "arm.rpi5"                        # May need to change
  key_pair    = var.key

  network {
    port = openstack_networking_port_v2.edge_public_port.id
  }

  user_data = <<-EOF
    #!/bin/bash
    sudo apt update
    sudo apt install -y python3-pip
    pip3 install fastapi uvicorn
  EOF
}

resource "openstack_networking_floatingip_v2" "edge_floating_ip" {
  provider = openstack.chiedge
  pool     = "public"
  port_id  = openstack_networking_port_v2.edge_public_port.id
}



