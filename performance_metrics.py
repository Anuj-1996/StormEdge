import logging

def format_value(value):
    if value >= 1e7:  # 1 crore (10 million)
        return f"Rs. {value / 1e7:.2f} Cr"
    elif value >= 1e5:  # 1 lakh (100 thousand)
        return f"Rs. {value / 1e5:.2f} L"
    elif value >= 1e3:  # 1 thousand
        return f"Rs. {value / 1e3:.2f} K"
    else:
        return f"Rs. {value:.2f}"

def evaluate_performance(results):
    if 'Portfolio' not in results.columns:
        logging.warning("Portfolio column not found in backtest results. Check backtester output.")
        return

    final_value = results['Portfolio'].iloc[-1]
    formatted_final_value = format_value(final_value)

    logging.info("Backtest complete. Final portfolio value: %s", formatted_final_value)
    
    initial_value = results['Portfolio'].iloc[0]
    formatted_initial_value = format_value(initial_value)

    abs_return = final_value - initial_value
    formatted_abs_return = format_value(abs_return)

    roi = (abs_return / initial_value) * 100

    logging.info("Initial Portfolio Value: %s", formatted_initial_value)
    logging.info("Absolute Return: %s", formatted_abs_return)
    logging.info("ROI: %.2f%%", roi)

    # Annualized ROI and CAGR
    try:
        start_date = results.index.min()
        end_date = results.index.max()
        num_days = (end_date - start_date).days
        num_years = num_days / 365.25

        if num_years > 0:
            annualized_roi = (final_value / initial_value) ** (1 / num_years) - 1
            cagr = annualized_roi

            logging.info("Annualized ROI: %.2f%%", annualized_roi * 100)
            logging.info("CAGR (Compounded Annual Growth Rate): %.2f%%", cagr * 100)
        else:
            logging.warning("Unable to calculate Annualized ROI and CAGR due to short date range.")
    except Exception as e:
        logging.error("Error calculating Annualized ROI and CAGR: %s", str(e))
