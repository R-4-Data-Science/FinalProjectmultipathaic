#' @export
RS_gui = function(){
  appDir = system.file("RS_int", package = "multipathaic")
  shiny::runApp(appDir, display.mode = "normal")
}
