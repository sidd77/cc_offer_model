drop table if exists analytics.SA_clicksData_cards;
create table analytics.SA_clicksData_cards as
  select distinct cs_p_id, trunc(cs_ed) as event_date, tm_product
  from autouk.auto_app_report_recommendation_apply
  where cs_ed between to_date('01-01-19', 'dd-mm-yy') and to_date('31-03-19', 'dd-mm-yy')
  union distinct
  select distinct cs_p_id, trunc(cs_ed) as event_date, tm_product
  from autouk.auto_app_report_recommendation_info
  where cs_ed between to_date('01-01-19', 'dd-mm-yy') and to_date('31-03-19', 'dd-mm-yy');

drop table if exists analytics.SA_searchData_cards;
create table analytics.SA_searchData_cards as
  select cs_p_id, cs_ed, apr, eligibility, rank, category, product_id, event_date,
  cashbacksavings,
  purchasesavings,
  btsavings
  from domo.prd_card_event
  where cs_ed between to_date('01-01-19', 'dd-mm-yy') and to_date('31-03-19', 'dd-mm-yy');

drop table if exists analytics.sa_score_cards;
create table analytics.sa_score_cards as
select cs_p_id, event_date, cr_credit_score_i, stdebt, ccdebt, cclimit, everdel,
       everdef, monthssincedel, monthssincedef, open_loans, open_creditcards
from analytics.creditreports_full
where event_date between to_date('01-01-19', 'dd-mm-yy') and to_date('31-03-19', 'dd-mm-yy');

drop table if exists analytics.sa_fintab_cards;
create table analytics.sa_fintab_cards as
with t1 as (select cs_p_id, max(cs_ed) as srch_date, event_date
            from analytics.SA_searchData_cards
            group by cs_p_id, event_date),
     t2 as (select * from analytics.SA_searchData_cards where concat(cs_p_id, cs_ed) in
            (select concat(cs_p_id, srch_date) from t1)),
     t3 as (select t2.*,
            cr_credit_score_i, stdebt, ccdebt, cclimit, everdel,
       everdef, monthssincedel, monthssincedef, open_loans, open_creditcards
            from t2 left join
            analytics.sa_score_cards as b
            on t2.cs_p_id =b.cs_p_id
            and date_trunc('month', t2.event_date) = date_trunc('month', b.event_date))
     select t3.*, a.tm_product from t3 left join analytics.SA_clicksData_cards as a on
     t3.cs_p_id=a.cs_p_id and t3.event_date=a.event_date and t3.product_id=a.tm_product;

drop table if exists analytics.sa_cards_data;
create table analytics.sa_cards_data as
  select apr, eligibility, rank, category,
         stdebt, ccdebt, cclimit, everdel,
         everdef, monthssincedel, monthssincedef,
         open_loans, open_creditcards,
         round((nvl(cashbacksavings,0)
                    + nvl(purchasesavings,0)
                    +  nvl(btsavings,0))/100, 0)
         as savings,
  Case
    When tm_product isNull Then 0
    Else 1
  END as clicked,
  Case
    When cr_credit_score_i < 280 then 'VP'
    When cr_credit_score_i < 380 and cr_credit_score_i > 279 then 'P'
    When cr_credit_score_i < 419 and cr_credit_score_i > 379 then 'F'
    When cr_credit_score_i < 466 and cr_credit_score_i > 418 then 'G'
    ELSE 'E'
  end as score,
  rank() over (partition by cs_p_id, event_date, category order by rank asc ) as rank1
  from analytics.sa_fintab_cards where not cr_credit_score_i isnull
  and cr_credit_score_i > 0 and eligibility > 0 and category <> 'Not Shown';
