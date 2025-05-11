output "floating_ip_out" {
  description = "Floating IP assigned to node1"
  value       = openstack_networking_floatingip_v2.floating_ip.address
}

output "edge_floating_ip_out" {
  description = "Floating IP assigned to CHI@Edge FastAPI node"
  value       = openstack_networking_floatingip_v2.edge_floating_ip.address
}
