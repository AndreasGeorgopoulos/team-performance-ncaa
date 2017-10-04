library(sandwich)           
library(lmtest)             
library(stargazer)
library(ggplot2)
library(relaimpo)
library(car)

# Import Final Dataset with Additional Variables (produced by python script)
test_bball <- read.csv("bball_data_readR.csv")


# Multicollinearity - VIF ---------------------------------------------
bball.model <- lm(outcome ~ Game + factor(Year) + same_conf + home_game + height_diversity + ethnic_diversity + region_diversity + school_diversity + degree_diversity + class_diversity + experience_with_opponent + experience_in_arena + conference_size + conf_wl + opp_games_played + opponent_win_loss + Coaching_Years + Coach_Win_Loss_Perc + Coach_NCAA_appear + Team_Comp_Change, data = test_bball)
alias( bball.model ) 
vifs<-vif(bball.model)
stargazer(vifs, type="html", title="Variance Inflation Factors",align=TRUE,omit.stat=c("LL","ser","f"), no.space=TRUE, out="vif.html")


# Add interaction terms and higher order terms:
test_bball$experience_in_arena_squared <- test_bball$experience_in_arena^2
test_bball$experience_with_opponent_squared <- test_bball$experience_with_opponent^2
test_bball$conference_size_squared <- test_bball$conference_size^2
test_bball$Coaching_Years_squared <- test_bball$Coaching_Years^2
test_bball$height_diversity_squared <- test_bball$height_diversity^2
test_bball$class_diversity_squared <- test_bball$class_diversity^2
test_bball$Coach_NCAA_appear_squared <- test_bball$Coach_NCAA_appear^2


# FORWARD-BACKWARD of Selected Predictors -----------------------------
library(leaps)
bball.model.null <- lm(outcome ~ 1, test_bball) 
bball.model.full <- lm(outcome ~ Game:Team_Comp_Change + Game:Coaching_Years + Game + home_game + height_diversity + log(ethnic_diversity) + class_diversity + class_diversity_squared + conference_size + conference_size_squared + conf_wl + opp_games_played + opponent_win_loss + Coaching_Years + Coaching_Years_squared + Coach_Win_Loss_Perc + Coach_NCAA_appear  + Team_Comp_Change, test_bball)
u<-step(bball.model.null, scope = list(lower = bball.model.null, upper = bball.model.full), direction = "both") 

# Best Model (min AIC)
bball.model.best.aic <- lm(outcome ~  home_game + height_diversity + log(ethnic_diversity)  + class_diversity + class_diversity_squared + conference_size + opp_games_played + opponent_win_loss + Coach_Win_Loss_Perc + Coach_NCAA_appear + Team_Comp_Change + Coaching_Years + Coaching_Years_squared, test_bball)
resettest(bball.model.best.aic, type = "fitted") # misspecification problem
summary(bball.model.best.aic)

# Final Model (max R-squared, AIC - AIC_model_min_aic <2 so ok)
bball.model.final <- lm(outcome ~  home_game + height_diversity + log(ethnic_diversity)  + class_diversity + class_diversity_squared + conference_size + conference_size_squared + conf_wl + opp_games_played + opponent_win_loss + Coach_Win_Loss_Perc + Coach_NCAA_appear + Team_Comp_Change + Coaching_Years + Coaching_Years_squared, test_bball)
summary(bball.model.final)
stargazer(bball.model.final, type="html", title="Regression Results by OLS",align=TRUE,omit.stat=c("LL","ser","f"), no.space=TRUE, out="regres_OLS.html")

# Residual plot
plot(bball.model.final)


# Check for Functional Form Misspecification-RESET Test ----------------
resettest(bball.model.final, type = "fitted") # reject at 5%--> no mispec problem


# HETEROSKEDASTICITY ---------------------------------------------------
# Plot Residuals
ggplot(bball.model.final, aes(.fitted, .resid)) + geom_point() + geom_hline(yintercept = 0)

# White test
fitted.bbal.final <- bball.model.final$fitted.values    
bptest(bball.model.final, ~ fitted.bbal.final + I(fitted.bbal.final^2))   # heteroskedasticity

# Robust errors
vcov.robust <- vcovHC(bball.model.final, "HC1")    
bball.model.final.robust <- coeftest(bball.model.final, vcov = vcov.robust)    
bball.model.final.robust
stargazer(bball.model.final.robust, type="html", title="Regression Results, Robust Errors",align=TRUE,omit.stat=c("LL","ser","f"), no.space=TRUE, out="regression_robust.htm")


# Percetnage contribution of each variable ----------------------------
calc.relimp(bball.model.final, rela = TRUE)
plot(calc.relimp(bball.model.final, rela = TRUE), las = 1)

axis(1, at = ,labels=c('home_game' , 'height_diversity' , 'log(ethnic_diversity)'  , 'class_diversity' , 'conference_size' , 'conference_size_squared' , 'conf_wl', 'opp_games_played' , 'opponent_win_loss' , 'Coach_Win_Loss_Perc', 'Coach_NCAA_appear' ,'Team_Comp_Change'), las = 2)

perc_contr = as.data.frame(calc.relimp(bball.model.final, rela = TRUE)$lmg)
names(perc_contr)[names(perc_contr)== 'calc.relimp(bball.model.final, rela = TRUE)$lmg'] <- "Value"
perc_contr['Variable'] <- rownames(perc_contr)
perc_contr <- perc_contr[order(perc_contr$Value, decreasing = TRUE),]
write.csv(perc_contr, "perc_contr.csv")


# Weighted Least Square Estimation -------------------------------------
res.u <- bball.model.final$residuals^2                                                    # save square resid of original model
aux.model <- lm(log(res.u) ~ fitted.bbal.final + I(fitted.bbal.final^2), test_bball)      # OLS square of residuals
g<-aux.model$fitted.values                                                                # compute g: fitted values of auxilary model
h <- exp(aux.model$fitted.values)                                                         # put exp(fitted ) to h

# Estimate original model by OLS by using weights 1/hi
bball.model.final.wls <- lm(outcome ~ home_game + height_diversity + log(ethnic_diversity)  + class_diversity + conference_size + conference_size_squared + conf_wl + opp_games_played + opponent_win_loss + Coach_Win_Loss_Perc + Coach_NCAA_appear + Team_Comp_Change,test_bball, weight = 1/h)
stargazer(maths.wls.model, type="html", title="Weighted Least Squares Regression Results",align=TRUE, dep.var.labels=c("Students satisfactory, 4th grade math (%)"), covariate.labels=c("Students eligible for free or reduced lunch (%)", "Logarithm form of school enrollment", "Logarithm form of expenditures per pupil"),omit.stat=c("LL","ser","f"), no.space=TRUE, out="hw3_1_wls.htm")


# Feasible Generalised Least Square Estimation --------------------------
aux.y <- log(bball.model.final$residuals^2)
aux.model <- lm(aux.y ~ home_game + height_diversity + log(ethnic_diversity)  + class_diversity + conference_size + conference_size_squared + conf_wl + opp_games_played + opponent_win_loss + Coach_Win_Loss_Perc + Coach_NCAA_appear + Team_Comp_Change, test_bball)
h <- exp(aux.model$fitted.values)

# Estimate original model by OLS by using weights 1/hi
bball.model.final.wls <- lm(outcome ~ home_game + height_diversity + log(ethnic_diversity)  + class_diversity + conference_size + conference_size_squared + conf_wl + opp_games_played + opponent_win_loss + Coach_Win_Loss_Perc + Coach_NCAA_appear + Team_Comp_Change,test_bball, weight = 1/h)

summary(bball.model.final.wls)
stargazer(bball.model.final.wls, type="html", title="WLS Regression Results, Robust Errors",align=TRUE, dep.var.labels=c("Students satisfactory, 4th grade math (%)"), covariate.labels=c("Students eligible for free or reduced lunch (%)", "Logarithm form of school enrollment", "Logarithm form of expenditures per pupil"),omit.stat=c("LL","ser","f"), no.space=TRUE, out="hw3_1wls_robust.htm")

# Get Robust Errors
coeftest(bball.model.final.wls, vcov = vcovHC(bball.model.final.wls, "HC1")) 
