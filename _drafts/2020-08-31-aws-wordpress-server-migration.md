---
layout: post
title:  "AWS Wordpress Server Migration"
date:   2020-08-24 11:11:11 -1800
categories: system-administration
---

Lightsail is a cloud platform that provides everything you need to deploy and host your WordPress website, including instances, managed databases, static IP addresses, and load balancers. Although we’ll be focusing on using Lightsail to launch your WordPress instance, you can also use Lightsail to deploy small-scale web applications, business software, developer sandboxes, and testing environments.

The Lightsail management console provides easy access to all the core AWS configuration options, so you can configure your server, your static IP addresses, and the DNS (Domain Name System) settings. 

It Has a Set Monthly Limit: With Lightsail, you pay an hourly rate for the resources you consume, up to a pre-arranged maximum monthly cost. 

It’s Flexible: What happens if your website outgrows its Lightscale subscription? If you require more resources, then you can upgrade your RAM and storage capacity at any time by migrating to a new Lightsail instance. Alternatively, if you need to cut costs or save resources, then you can switch to a smaller Lightsail instance. 

If your WordPress website is business-critical or you plan to implement advanced or complex functionality, then you should either plan to purchase an additional support package or opt for an alternative platform that provides technical support as standard. 


- Login to console

- Navigate to Services > Lightsail

- Create instance: Linux (OS only) Debian 9.5

- Create Static IP > assign to new instance

- Click on the instance and go to Networking Tab

- Add rules for SSH, HTTP, HTTPS, MySQL/Aurora (and PostgreSQL if needed) and click Save

- Go back to the Connect Tab and click `Connect using SSH`

- Transfer your DNS settings to point to the new site