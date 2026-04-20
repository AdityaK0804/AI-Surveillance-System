// Middleware to check if user is authenticated
export function isAuthenticated(req, res, next) {
    if (req.session && req.session.userId) {
        return next();
    }
    res.redirect("/auth/login");
}

// Middleware to pass user data to all views
export function setUserLocals(req, res, next) {
    res.locals.isLoggedIn = !!(req.session && req.session.userId);
    res.locals.userName = req.session?.userName || null;
    res.locals.userEmail = req.session?.userEmail || null;
    next();
}
