---
layout: post
title:  "Thinking Outside the Lego Box"
date:   2019-12-07 10:23:47 -0800
categories: datascience
---

## The Lego Model and Other Metaphysical Approaches to Data Science

**TODO**
* Clean up the hideous default `Jekyll` markdown formatting of this post
* Add some pictures
* Rewrite the entire post because I originally wrote it as a stream of consciousness too late at night and I forgot to say anything about legos, which was the intended topic...Oops.

## Box #1: Database Tables

1. Unpacking

The project starts out with a giant box containing many smaller boxes, each of which contains thousands of individual data points. Your first task is to familiarize yourself with what each smaller box contains, noting the characteristics of each group, as well as how the groups relate to each other. These are the relationships that are already visible and known. Your job, however, is to determine new, unknown relationships that might be possible between some contents of only some of the various boxes. 

2. Mixing and Measuring

You also have at your disposal several measuring cups, that is, tools for measuring other characteristics of the boxes that will you help you uncover some clues about hidden relationships between individual pieces of data. Mathematically speaking, these are the statistical relationships such as the mean, variance, standard deviation, maximum and minimum, to name a few. 

---

## Box #2: Domain

1. Outside In vs Inside Out

One of the main reasons companies hire a data scientist is to discover something they don’t already know based on information they already have. A common trap employees at most companies fall into is thinking too much from the inside looking out. They spend all their time thinking about the products they sell, and how to sell more of those products to people outside (customers). 

2. Finding the Union: Customer Values and Company Mission

The business processes currently place are set up to do this in a certain way based on current assumptions about what will achieve sales. 
However, the customers outside don’t care about any of this, obviously, and the entire marketing budget is spent basically trying to figure what it is they do actually care about and why. The only means of determining what customers do actually care about (apart from asking them directly, which is futile because the answers they give are unreliable for reasons I won't get into here) - as it relates to the company’s products, exists in past sales metrics. 

3. Predicting Preferences 
It’s not enough to know that 500 customers buy Product A every year, while another 500 buy Product B. We have to look at what characteristics the individuals of each group have in common, and how that might have a relationship with the characteristics of the product they buy in order to make assumptions about WHY they buy that one and not the other. To make it more complicated, it would be great to know why they buy this company’s product, instead of a competitor’s similar offer. 

4. Probabilities and Bayesian Stats

The way we go about inferring any of these hidden relationships is by entering the world of Probabilities and Bayesian statistics in particular. 

---

## Box #3: System

1. The Meta Box: Problem > Solution

The biggest box of the project is the one you didn’t realize you’re standing in yourself, along with all the data and the client that sent it to you to analyze. It’s a bit invisible meta-box that has everyone focused on finding an answer to a specific question. The problem here is that all your work is being done based on the assumption that you’re asking the right question. The hypothesis you formulate will be based on the question being asked, and the question itself is based on an assumption about the existence of a bigger problem that needs to be solved. If a company’s primary focus is on the bottom-line (which most tend to be), the question, the hypothesis, and the results will all be driven by solutions that relate directly to revenues, margins, costs, velocity, etc. This makes it easy to miss opportunities for value-creation which might only be detectable by looking at qualitative and categorical data. 

2. The Invisible Unquantifiables

There are more nuanced considerations which can be difficult to quantify such as brand, as well as the psychological and behavioral elements of customers not directly visible in their demographics. Take two identical products, each sold by two different companies and you’ll be amazed at how different people exhibit a clear distinction in preference for one over the other - the only actual difference the two products have is in the logo slapped on each one. The logos have different colors and different wordmarks - but it’s not as simple as customer 1 likes red and unknowingly chose the product with the red logo basically only because of this color preference. That could actually be the case - but more often, customers just tend to like one brand because it’s familiar to them, and I could write an entire article about what “familiarity” really means because it is itself extremely nuanced - briefly two examples could be “this is what my grandpa always bought” and now you have brand trust based on sentimental values and family tradition. It could be crap, but it’s the crap they know and love. Try to get them to buy a new and improved one, and you will be heavily resisted for no reason that can explained rationally. The more obvious example is based on rational things like reliability, quality, ease, efficiency, and other characteristics the customer has come to associate with the brand and its products overall based on experience. Where’s the dataset for all of this? Good question. 

3. Questioning How You Question: Rethinking System Design 

If you thought I was going to provide an answer in this 3rd box, you forgot the whole reason I added this one, which is to make a point about focusing more on exploring the questions themselves rather than finding a solution that may or may not be the one you actually needed if your question was misguided to begin with. If this seems overly philosophical, that’s because the box we get stuck in is the assumption about the overall system we have already built for us. It’s a valuable if not annoying question that needs to be asked and usually is avoided - is the system we have for collecting and analyzing all this data the best system possible, and if we were to create a new system from scratch to increase the reliability of our results, what would that system look like? Such a question represents a fundamental line where data science and software architecture link together.