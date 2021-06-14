# FTC-copter

## plant 
### quadcopter 
> linear plant
>> small angle assumption
>> 
>> use Euler angle
>> 
> nonlinear plant
>> use quat, dcm

-------------------

## Controller 
> integral sliding mode
> 
> Adaptive Integral sliding mode
> 
> pd controller (to make phi, theta reference)


-----------------

## code file
### plant
```
copter.py
```

### Controller
+ Integral sliding mode controller
```
ISMC.py
```
+ Adaptive sliding mode controller
```
AdapticeISMC.py
```

