#
# Multi-Path AIC Selection - Interactive Shiny App
# Demonstrates the multipathaic package with visualizations
#

library(shiny)
library(multipathaic)
library(ggplot2)
library(plotly)
library(DT)

# UI Definition
ui <- fluidPage(
  theme = bslib::bs_theme(bootswatch = "flatly"),
  
  titlePanel("Multi-Path AIC Selection Explorer"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      
      h4("Data Setup"),
      
      selectInput("family", "Model Family:",
                  choices = c("Gaussian (Linear)" = "gaussian",
                              "Binomial (Logistic)" = "binomial"),
                  selected = "gaussian"),
      
      numericInput("n_obs", "Number of Observations:", 
                   value = 150, min = 50, max = 500, step = 10),
      
      numericInput("n_pred", "Number of Predictors:", 
                   value = 8, min = 4, max = 20, step = 1),
      
      hr(),
      
      h4("Algorithm Parameters"),
      
      numericInput("K", "Max Steps (K):", 
                   value = 8, min = 3, max = 15, step = 1),
      
      numericInput("delta", "AIC Tolerance (δ):", 
                   value = 2, min = 0, max = 5, step = 0.5),
      
      numericInput("L", "Max Models per Step (L):", 
                   value = 50, min = 10, max = 100, step = 10),
      
      numericInput("B", "Stability Resamples (B):", 
                   value = 30, min = 10, max = 100, step = 10),
      
      numericInput("Delta", "Plausibility Tolerance (Δ):", 
                   value = 2, min = 0, max = 5, step = 0.5),
      
      numericInput("tau", "Stability Threshold (τ):", 
                   value = 0.6, min = 0, max = 1, step = 0.1),
      
      hr(),
      
      actionButton("run", "Run Analysis", 
                   class = "btn-primary btn-lg btn-block"),
      
      br(), br(),
      
      downloadButton("download_report", "Download Report", 
                     class = "btn-success btn-block")
    ),
    
    mainPanel(
      width = 9,
      
      tabsetPanel(
        id = "tabs",
        
        # Tab 1: Overview
        tabPanel("Overview",
                 br(),
                 uiOutput("overview_ui"),
                 br(),
                 h4("Variable Stability Scores"),
                 plotlyOutput("stability_plot", height = "400px"),
                 br(),
                 h4("Models by Step"),
                 plotOutput("models_by_step", height = "300px")
        ),
        
        # Tab 2: Plausible Models
        tabPanel("Plausible Models",
                 br(),
                 h4("Selected Plausible Models"),
                 DTOutput("plausible_table"),
                 br(),
                 h4("Model Overlap Heatmap"),
                 plotlyOutput("overlap_heatmap", height = "500px"),
                 br(),
                 h4("Variable Inclusion Probabilities"),
                 plotlyOutput("inclusion_plot", height = "400px")
        ),
        
        # Tab 3: Branching Tree
        tabPanel("Branching Visualization",
                 br(),
                 h4("Multi-Path Branching by Step"),
                 plotlyOutput("branching_tree", height = "600px"),
                 br(),
                 uiOutput("branching_info")
        ),
        
        # Tab 4: Diagnostics
        tabPanel("Diagnostics",
                 br(),
                 h4("Model Performance"),
                 uiOutput("performance_ui"),
                 br(),
                 conditionalPanel(
                   condition = "input.family == 'binomial'",
                   h4("Confusion Matrix"),
                   plotOutput("confusion_plot", height = "400px"),
                   br(),
                   verbatimTextOutput("confusion_metrics")
                 )
        ),
        
        # Tab 5: About
        tabPanel("About",
                 br(),
                 h3("Multi-Path AIC Selection"),
                 p("This Shiny app demonstrates the", code("multipathaic"), 
                   "R package for multi-path forward selection using AIC."),
                 
                 h4("Key Features"),
                 tags$ul(
                   tags$li("Explores multiple competitive model paths simultaneously"),
                   tags$li("Assesses variable stability via bootstrap resampling"),
                   tags$li("Identifies plausible models balancing fit and stability"),
                   tags$li("Supports both Gaussian and binomial regression")
                 ),
                 
                 h4("Algorithms"),
                 tags$ol(
                   tags$li(strong("build_paths()"), " - Multi-path forward selection with branching"),
                   tags$li(strong("stability()"), " - Bootstrap stability estimation"),
                   tags$li(strong("plausible_models()"), " - AIC + stability filtering"),
                   tags$li(strong("multipath_aic()"), " - Complete pipeline")
                 ),
                 
                 h4("Installation"),
                 pre('remotes::install_github("R-4-Data-Science/FinalProjectmultipathaic")'),
                 
                 h4("Authors"),
                 p("Michael Obuobi, Jinchen Jiang, Far Rahmati"),
                 p("Auburn University, 2025"),
                 
                 hr(),
                 p("Repository:", 
                   a("GitHub", 
                     href = "https://github.com/R-4-Data-Science/FinalProjectmultipathaic",
                     target = "_blank"))
        )
      )
    )
  )
)

# Server Logic
server <- function(input, output, session) {
  
  # Reactive values to store results
  results <- reactiveValues(
    data = NULL,
    result = NULL,
    run_time = NULL
  )
  
  # Generate data and run analysis
  observeEvent(input$run, {
    
    # Show progress
    withProgress(message = 'Running Multi-Path AIC...', value = 0, {
      
      # Generate synthetic data
      incProgress(0.1, detail = "Generating data...")
      set.seed(123)
      n <- input$n_obs
      p <- input$n_pred
      X <- as.data.frame(matrix(rnorm(n*p), n, p))
      names(X) <- paste0("x", 1:p)
      
      # Generate response
      if (input$family == "gaussian") {
        # Linear: make first 3 variables important
        beta <- c(2, -1.5, 1, rep(0, p-3))
        y <- as.numeric(as.matrix(X) %*% beta + rnorm(n, 1))
      } else {
        # Logistic: make first 3 variables important
        n_important <- min(3, p)
        coefs <- rep(0, p)
        coefs[1:n_important] <- c(1.5, -2, 1)[1:n_important]
        eta <- as.numeric(as.matrix(X) %*% coefs)
        prob <- 1 / (1 + exp(-eta))
        y <- rbinom(n, 1, prob)
      }
      
      results$data <- list(X = X, y = y)
      
      # Run multi-path AIC
      incProgress(0.2, detail = "Building paths...")
      start_time <- Sys.time()
      
      result <- multipath_aic(
        X = X,
        y = y,
        family = input$family,
        K = input$K,
        eps = 1e-6,
        delta = input$delta,
        L = input$L,
        B = input$B,
        resample_fraction = 0.8,
        Delta = input$Delta,
        tau = input$tau,
        verbose = FALSE
      )
      
      end_time <- Sys.time()
      results$run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      incProgress(0.7, detail = "Finalizing results...")
      results$result <- result
      
      incProgress(1, detail = "Complete!")
    })
    
    showNotification("Analysis complete!", type = "message", duration = 3)
  })
  
  # Overview UI
  output$overview_ui <- renderUI({
    req(results$result)
    
    result <- results$result
    n_models <- nrow(result$plaus$plausible_models)
    n_steps <- length(result$forest$path_forest$frontiers)
    best_aic <- round(result$plaus$best_aic, 2)
    
    tagList(
      fluidRow(
        column(3, 
               div(class = "card bg-primary text-white",
                   div(class = "card-body",
                       h3(n_steps),
                       p("Steps Explored")
                   )
               )
        ),
        column(3,
               div(class = "card bg-success text-white",
                   div(class = "card-body",
                       h3(n_models),
                       p("Plausible Models")
                   )
               )
        ),
        column(3,
               div(class = "card bg-info text-white",
                   div(class = "card-body",
                       h3(best_aic),
                       p("Best AIC")
                   )
               )
        ),
        column(3,
               div(class = "card bg-warning text-white",
                   div(class = "card-body",
                       h3(round(results$run_time, 1), "s"),
                       p("Run Time")
                   )
               )
        )
      )
    )
  })
  
  # Stability plot
  output$stability_plot <- renderPlotly({
    req(results$result)
    
    pi <- results$result$stab$pi
    df <- data.frame(
      Variable = names(pi),
      Stability = as.numeric(pi)
    )
    df <- df[order(df$Stability, decreasing = TRUE), ]
    df$Variable <- factor(df$Variable, levels = df$Variable)
    
    p <- ggplot(df, aes(x = Variable, y = Stability)) +
      geom_col(aes(fill = Stability)) +
      geom_hline(yintercept = input$tau, linetype = "dashed", color = "red") +
      scale_fill_gradient(low = "lightblue", high = "darkblue") +
      labs(title = "Variable Stability Across Resamples",
           x = "Variable", y = "Stability Score (π)",
           caption = paste("Red line = τ threshold (", input$tau, ")")) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggplotly(p)
  })
  
  # Models by step
  output$models_by_step <- renderPlot({
    req(results$result)
    
    frontiers <- results$result$forest$path_forest$frontiers
    n_models <- sapply(frontiers, nrow)
    
    df <- data.frame(
      Step = 1:length(n_models),
      Models = n_models
    )
    
    ggplot(df, aes(x = Step, y = Models)) +
      geom_line(color = "steelblue", size = 1) +
      geom_point(color = "steelblue", size = 3) +
      labs(title = "Number of Models Retained at Each Step",
           x = "Forward Selection Step", y = "Number of Models") +
      theme_minimal() +
      theme(text = element_text(size = 12))
  })
  
  # Plausible models table
  output$plausible_table <- renderDT({
    req(results$result)
    
    plaus <- results$result$plaus$plausible_models
    
    display_df <- data.frame(
      Model_ID = seq_len(nrow(plaus)),
      AIC = round(plaus$AIC, 2),
      Variables = sapply(plaus$model, function(x) paste(x, collapse = ", ")),
      Avg_Stability = round(plaus$avg_stability, 3),
      N_Vars = sapply(plaus$model, length)
    )
    
    datatable(display_df, 
              options = list(pageLength = 10, scrollX = TRUE),
              rownames = FALSE)
  })
  
  # Overlap heatmap
  output$overlap_heatmap <- renderPlotly({
    req(results$result)
    
    plaus <- results$result$plaus$plausible_models
    
    if (nrow(plaus) < 2) {
      return(plotly_empty() %>% 
               layout(title = "Need at least 2 plausible models for overlap heatmap"))
    }
    
    # Create overlap matrix
    n_models <- min(nrow(plaus), 15)  # Limit to 15 for visibility
    plaus_subset <- plaus[1:n_models, ]
    
    overlap_matrix <- matrix(0, n_models, n_models)
    
    for (i in 1:n_models) {
      for (j in 1:n_models) {
        vars_i <- plaus_subset$model[[i]]
        vars_j <- plaus_subset$model[[j]]
        
        if (length(vars_i) == 0 || length(vars_j) == 0) {
          overlap_matrix[i, j] <- 0
        } else {
          overlap <- length(intersect(vars_i, vars_j)) / 
            length(union(vars_i, vars_j))
          overlap_matrix[i, j] <- overlap
        }
      }
    }
    
    model_labels <- paste("Model", 1:n_models)
    
    plot_ly(
      x = model_labels,
      y = model_labels,
      z = overlap_matrix,
      type = "heatmap",
      colors = colorRamp(c("white", "orange", "red")),
      hovertemplate = 'Model %{x}<br>Model %{y}<br>Overlap: %{z:.2f}<extra></extra>'
    ) %>%
      layout(
        title = "Jaccard Overlap Between Plausible Models",
        xaxis = list(title = ""),
        yaxis = list(title = "")
      )
  })
  
  # Inclusion probability plot
  output$inclusion_plot <- renderPlotly({
    req(results$result)
    
    inclusion <- results$result$plaus$inclusion
    df <- data.frame(
      Variable = names(inclusion),
      Inclusion = as.numeric(inclusion)
    )
    df <- df[order(df$Inclusion, decreasing = TRUE), ]
    df$Variable <- factor(df$Variable, levels = df$Variable)
    
    p <- ggplot(df, aes(x = Variable, y = Inclusion)) +
      geom_col(aes(fill = Inclusion)) +
      geom_hline(yintercept = 1, linetype = "dashed", color = "darkgreen") +
      scale_fill_gradient(low = "lightgreen", high = "darkgreen") +
      labs(title = "Variable Inclusion Across Plausible Models",
           x = "Variable", y = "Inclusion Probability") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggplotly(p)
  })
  
  # Branching tree visualization
  output$branching_tree <- renderPlotly({
    req(results$result)
    
    frontiers <- results$result$forest$path_forest$frontiers
    
    # Build tree data
    nodes <- list()
    edges <- list()
    node_id <- 0
    
    # Root node
    nodes[[1]] <- list(id = node_id, step = 0, aic = results$result$forest$all_models$AIC[1], 
                       vars = "", parent = -1)
    parent_map <- list()
    parent_map[[""]] <- 0
    
    for (step in seq_along(frontiers)) {
      frontier <- frontiers[[step]]
      
      for (i in 1:min(nrow(frontier), 20)) {  # Limit for visualization
        node_id <- node_id + 1
        vars <- frontier$model[[i]]
        key <- frontier$key[i]
        
        # Find parent (model with one less variable)
        parent_vars <- if (length(vars) > 1) {
          sapply(vars, function(v) {
            parent_v <- setdiff(vars, v)
            paste(sort(parent_v), collapse = "|")
          })
        } else {
          ""
        }
        
        parent_id <- parent_map[[""]]  # Default to root
        for (pv in parent_vars) {
          if (!is.null(parent_map[[pv]])) {
            parent_id <- parent_map[[pv]]
            break
          }
        }
        
        nodes[[node_id + 1]] <- list(
          id = node_id,
          step = step,
          aic = frontier$AIC[i],
          vars = paste(vars, collapse = ", "),
          parent = parent_id
        )
        
        parent_map[[key]] <- node_id
        
        edges[[length(edges) + 1]] <- list(from = parent_id, to = node_id)
      }
    }
    
    # Convert to data frames
    nodes_df <- do.call(rbind, lapply(nodes, function(n) {
      data.frame(id = n$id, step = n$step, aic = n$aic, 
                 vars = n$vars, stringsAsFactors = FALSE)
    }))
    
    # Create sunburst-style plot
    plot_ly() %>%
      add_trace(
        type = "scatter",
        x = nodes_df$step,
        y = nodes_df$aic,
        text = paste("Step:", nodes_df$step, "<br>AIC:", round(nodes_df$aic, 2),
                     "<br>Vars:", nodes_df$vars),
        mode = "markers",
        marker = list(
          size = 8,
          color = nodes_df$step,
          colorscale = "Viridis",
          showscale = TRUE,
          colorbar = list(title = "Step")
        ),
        hovertemplate = '%{text}<extra></extra>'
      ) %>%
      layout(
        title = "Multi-Path Branching Tree (Step vs AIC)",
        xaxis = list(title = "Forward Selection Step"),
        yaxis = list(title = "AIC")
      )
  })
  
  # Branching info
  output$branching_info <- renderUI({
    req(results$result)
    
    frontiers <- results$result$forest$path_forest$frontiers
    total_models <- sum(sapply(frontiers, nrow))
    
    tagList(
      p(strong("Total unique models explored:"), total_models),
      p(strong("Final step:"), length(frontiers)),
      p("Each point represents a model. Color indicates the step at which it was generated.")
    )
  })
  
  # Performance UI
  output$performance_ui <- renderUI({
    req(results$result)
    
    plaus <- results$result$plaus$plausible_models
    
    if (nrow(plaus) == 0) {
      return(p("No plausible models found."))
    }
    
    best_vars <- plaus$model[[1]]
    best_aic <- plaus$AIC[1]
    best_stability <- plaus$avg_stability[1]
    
    tagList(
      h5("Best Model (Lowest AIC):"),
      p(strong("Variables:"), paste(best_vars, collapse = ", ")),
      p(strong("AIC:"), round(best_aic, 2)),
      p(strong("Average Stability:"), round(best_stability, 3)),
      p(strong("Number of Variables:"), length(best_vars))
    )
  })
  
  # Confusion plot (for binomial)
  output$confusion_plot <- renderPlot({
    req(results$result)
    req(input$family == "binomial")
    
    plaus_set <- results$result$plaus$plausible_models
    
    if (nrow(plaus_set) == 0) return(NULL)
    
    # Get confusion matrix data
    selected_vars <- plaus_set$model[[1]]
    X <- results$result$forest$model_data$X
    y <- results$result$forest$model_data$y
    
    df <- data.frame(y = y, X)
    form <- as.formula(paste("y ~", paste(selected_vars, collapse = " + ")))
    model <- glm(form, data = df, family = binomial())
    
    p_hat <- predict(model, type = "response")
    y_pred <- ifelse(p_hat >= 0.5, 1, 0)
    
    TP <- sum(y_pred == 1 & y == 1)
    TN <- sum(y_pred == 0 & y == 0)
    FP <- sum(y_pred == 1 & y == 0)
    FN <- sum(y_pred == 0 & y == 1)
    
    conf_matrix <- matrix(c(TP, FP, FN, TN), nrow = 2, byrow = TRUE)
    
    # Plot confusion matrix
    conf_df <- data.frame(
      Predicted = rep(c("1", "0"), each = 2),
      Actual = rep(c("1", "0"), 2),
      Count = c(TP, FP, FN, TN)
    )
    
    ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Count)) +
      geom_tile(color = "white", size = 2) +
      geom_text(aes(label = Count), size = 20, color = "white") +
      scale_fill_gradient(low = "lightblue", high = "darkblue") +
      labs(title = "Confusion Matrix (Cutoff = 0.5)",
           x = "Actual Class", y = "Predicted Class") +
      theme_minimal() +
      theme(text = element_text(size = 14))
  })
  
  # Confusion metrics
  output$confusion_metrics <- renderPrint({
    req(results$result)
    req(input$family == "binomial")
    
    if (nrow(results$result$plaus$plausible_models) > 0) {
      confusion_metrics(results$result, model_index = 1, cutoff = 0.5, verbose = TRUE)
    }
  })
  
  # Download report
  output$download_report <- downloadHandler(
    filename = function() {
      paste0("multipathaic_report_", Sys.Date(), ".html")
    },
    content = function(file) {
      # Create a simple HTML report
      req(results$result)
      
      html_content <- paste0(
        "<html><head><title>Multi-Path AIC Report</title></head><body>",
        "<h1>Multi-Path AIC Selection Report</h1>",
        "<h2>Parameters</h2>",
        "<ul>",
        "<li>Family: ", input$family, "</li>",
        "<li>Observations: ", input$n_obs, "</li>",
        "<li>Predictors: ", input$n_pred, "</li>",
        "<li>K: ", input$K, "</li>",
        "<li>Delta: ", input$Delta, "</li>",
        "<li>Tau: ", input$tau, "</li>",
        "</ul>",
        "<h2>Results</h2>",
        "<p>Plausible Models: ", nrow(results$result$plaus$plausible_models), "</p>",
        "<p>Best AIC: ", round(results$result$plaus$best_aic, 2), "</p>",
        "</body></html>"
      )
      
      writeLines(html_content, file)
    }
  )
}

# Run the application
shinyApp(ui = ui, server = server)