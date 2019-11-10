---
layout: post
title:  "So You Think You're a Data Scientist"
date:   2019-11-07 10:23:47 -0800
categories: projects datascience
---

# So You Think You're a Data Scientist?

If you had asked me four weeks ago to define 'linear regression' or how it can be useful for planning a business strategy or financial investment, I probably would have just stared at you blankly and then tried to make something up that we'd both know was BS. I knew a little about statistics and I had a conceptual grasp of predictive analytics (I've worked in marketing for 10 years, c'mon) but I was very, very far away from being an expert in either arena.

> Or so I thought...

One short month ago marked the beginning of my journey toward becoming a professional data scientist. I had been teaching myself to code since I started designing websites (about 5 years ago), and I had taken a data analysis class with General Assembly to bolster my marketing / web analytics wheelhouse, so maybe the possibility of me suddenly deciding to become a software engineer shouldn't have surprised anyone.

As a front-end web developer I quickly got bored with `html` and `css` and started playing around with `php` and `javascript`. Moving all my website clients over to AWS (because DUH), it wasn't long before I was configuring and managing linux/LAMP servers in my sleep. Learning the basics of database administration (mostly MySql because, let's face it, `wordpress` is awesome now) got me interested in the actual hardware and I wanted to know about everything that was going on under the hood. So for my 30th birthday, I [built my own supercomputer](./projects/pc-build.html) from the ground up with parts I bought online (well, my parents bought them but that's besides the point. But..Thanks Mom and Dad! You guys are the best.).

However, while all this was going on, I was also the co-founder of a small marketing agency [kinetik](https://kinetik.la) so I couldn't dedicate as much time as I wanted to toward tech. I was getting burnt out on marketing. And my life in general. 32 years old and deep down the only sure thing I knew about the world was that I hated my life, didn't feel that connected to anyone, and plain and simple, I didn't like how I spent my time during the day. There were a lot of days I started to sleep as long as I could just to get through them. Other days I couldn't sleep at all. I knew I wasn't fulfilling my destiny, and whatever path(s) I had been following for the last three decades were not the right ones. I felt depressed and anxious almost all of the time.  I felt...lost.

There was no over night 'aha!' epiphany. It was like molasses getting poured out of a jar. Except this particular molasses is afraid to leave the jar because somewhere along the way of its miserable existence, it kind of just got used to being stuck in a place where it can't breath or do or be anything but sh*tty old molasses; so the idea of suddenly being outside the jar is absolutely terrifying. However, I also ran out of money, several times, over and over again. So, at the behest of my investors (same ones who paid for the computer parts), I basically had no choice but to hand everything over to my business partner and quit the company.

Giving up on Kinetik was one of the most difficult and painful decisions I have ever had to make in my life. Somehow giving up on your dream is much harder than struggling for two and a half years to keep it alive. But the dream was over, and it was time to wake up.

I started studying for the COMPTIA A+ exam, and took an online course for Python. I LOVED IT. Finally, I was ready to push the `reset` button on career, and on my entire life. So I pushed it. 'No more Molasses!'

That's when everything changed.

---

I'm now 4 weeks in to a full-time data science bootcamp, and just completed my first big project. If you were to ask me today if I know anything about 'linear regression', I'd probably just show you instead of tell you. And that's I'm going to do. Because today, unlike 4 weeks ago, I know how take over 25,000 data points subdivided into 18 different variables, and create a prediction model for increasing the property value of a house in the greater Seattle area. And I can say that whatever predictions are made using my model will carry 78% certainty in being correct.  I can tell you what the 3 most important variables are for house prices, and show you with pretty colored maps why I'm right. And that's what this blog post is really supposed to be about. So let's get going.

---



# Predicting Home Values with Multiple Linear Regression

The goal of this first project was to identify best combination of variable(s) for predicting property values in King County, Washington, USA.
