Hello, friends, in this lecture video, we look at a more elegant solution of the problem that we

are trying to solve, and for that we'll be using the event module, which we have previously discussed.

So this is the code that we discussed in the last lecture video.

So we'll obviously not be needing this.

All that we need to do is find a way to let our Python interpreter know the exact amount of time to

wait before moving on to execute the next command.

But to do that, we need to know when exactly is the historical data output completed for a given figure.

So like, you know, it's a WebSocket connection.

Candles are coming to us one by one.

So, for example, if you want five thousand candles for a given digger, W.S. will stream the candles

to us.

But then we need to know at what point has the last candle for that particular tiger has been given

to us?

And thankfully, W.S. does provide us a way to get that information.

So in this case, if you have used the wrapper function, historical data.

So let me take you back to historical data documentation and we will see what another function we can

use to get around this problem.

All right.

So we are in the documentation of the historical data.

And you can see that there are two functions.

One is historical data, which we have been using, and we are fairly familiar with this particular

function.

But then there is another yapper function.

And I kind of ignored this particular Erap or function in the introductory course because we had no

use of this particular function there.

But if we want to use the event module, we will have to use historical data and function.

Let me tell you that in addition to historical data, there are a lot of other API calls for which W.S.

also has some kind of an end function.

And that is very useful because once you have the end function, you can manipulate that particular

function and set the relevant event while the data is being streamed.

While you are receiving the candle, it will happen using the historical data wrapper function.

But as soon as the last candle is given to you, historical data and wrapper function will get triggered.

And within this proper function, you can set the event or you can clear the flag for that particular

event.

So first of all, let me copy this particular function and let us include it in our trading application.

So we are defining historical data and we need Salvini request ID and then it also didn't start and

end, which pertains to the start time and the end time.

So let us copy these two lines as well.

So we have copied over the historical data and function, but this is obviously not the wrapper function

that is going to help us because all that we are doing in this particular app or function is printing

out or announcing that we have finished extracting the historical data for a given request.

We don't want to do just that.

We want to do something else.

So we will revisit this particular wrapper function.

But in the meantime, why don't we do the most important thing, which is creating an event so we all

know how to create an event.

So let's call this event event.

You can call it historical data event or whatever.

This is just a variable name.

And then here we need to use the event module of threading library.

So this will create an object of event class, which is going to be very beneficial for us to keep track

of when an event occurs and when that event occurs, what we want to do.

So first of all, as soon as this particular command is run, which means as soon as we run the request

historical data Ekland function, we want to wait.

So we'll be using something called event rate.

We were earlier giving time to sleep in this case here is going to be even godwit.

And what does this command do?

All of this command is telling the Python interpreter is to wait till the events flag is set.

So right now we have just created an event object.

We have not set the flag to anything by default.

There is no flag set.

So what we are saying here is that until unless the flag for the event is not said keyboarding, do

not go any further.

And where do you think we'll be setting the flag for this particular event?

Well, you guessed it right in the historical data and grapple function.

So here, as soon as the historical data is entered, we know that we want to move on to the next picture.

And therefore, in this case, we are going to say event dot set.

OK, so now that the flag is set, this loop can continue executing and the loop can then go to the

next picture and repeat executing this particular Ekland function for the second ticket.

Let me again explain what will happen here.

You will run this particular function for the first Tyga Facebook.

As soon as you will run this function request historical data line function will get triggered and you

will start receiving data through historical data about function.

So you will get the first second Bertaud bar after you finish receiving five thousand or whatever number

of bars you need.

Historical data and.

Proper function will get triggered automatically and once this will get triggered, that means the data

extraction has been completed for that particular trigger.

And at this point, we set the event flag.

And what will happen if we set this event flag this way?

It will stop.

Now, your python interpreter can move on and continue executing this particular loop.

So now it will go to the second degree, which is Amazon.

And then again, it will execute the request historical data Ekland function for Amazon.

But what will happen next?

Will there be any way to know?

The flag is already said.

We had said the flag here, so there'll be no waiting anymore.

This will continue executing.

So in this case we'll get an error before getting Amazon's data will jump on to Intel and before finishing

Intel's data will jump onto data frame function and we'll get an error.

So what do we do?

Well, we need to clear the flag before executing the second Eggland function.

And that's what we are going to do here.

All that we need to do here is event not clear before executing the request.

Historical data eggland function for the second tyga.

We want to clear the flag so that the event rate command makes sense again.

We ensure that the Python interpreter waits for the event to be set again.

All right, so we clear the flag before executing this call and this is pretty much it.

This is the more elegant way of doing things.

So let us now run the record and let us see how long it takes to run and whether we get any errors or

not.

OK, so it's running.

So you can see that we are getting the data and it is done.

We have got data for all the three tickers and as soon as the statement is entered, we are manipulating

the event and you can see that we have got the data.

OK, so this is how I want you to extract historical data going forward, or for that matter, if there

is any API call that is going to take a long time and you don't really know how much time you need to

wait before executing the next command.

I suggest using the event module and you will see that I am going to make extensive use of event module

later in this course when we implement our strategy, because we have to do a number of things simultaneously

and we need to ensure that we execute certain commands only after we have a confirmation that the previous

command has been executed and an event has already occurred.

So this lecture should be a good practice, a good heads up for you because you would be seeing more

of this.

So let me know if you have any questions, and I'm more than happy to answer.

Thanks a lot.