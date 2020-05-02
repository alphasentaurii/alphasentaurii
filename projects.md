---
layout: page
title: projects
---

# `projects`

<h1>Latest Projects</h1>

<ul>
  {% for post in site.projects %}
    <li>
      <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
      <p>{{ post.excerpt }}</p>
    </li>
  {% endfor %}
</ul>


## coding

* [`python`](/projects/python.html)

* [`datascience`](/projects/datascience.html)

* [`webdesign`](/projects/webdesign.html)

* * *

## tech

* [`pc builds`](/projects/pc-builds.html)

* [`linux`](/projects/linux.html)
