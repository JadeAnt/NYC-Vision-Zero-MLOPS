variable "suffix" {
  description = "Suffix for resource names (use project1)"
  type        = string
  nullable = false
}

variable "key" {
  description = "Name of key pair"
  type        = string
  default     = "nyc-vision-zero"
}

variable "nodes" {
  type = map(string)
  default = {
    "node1" = "192.168.1.11"
    "node2" = "192.168.1.12"
    "node3" = "192.168.1.13"
  }
}
