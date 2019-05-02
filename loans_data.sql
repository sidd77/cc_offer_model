drop table if exists analytics.SA_clicksData_loans;
create table analytics.SA_clicksData_loans as
  select distinct cs_p_id, cs_ed, trunc(cs_ed) as event_date, rank, category, tm_product
  from autouk.auto_app_product_loan_click
  where cs_ed between to_date('01-01-19', 'dd-mm-yy') and to_date('31-03-19', 'dd-mm-yy')
  and not rank isnull and rank <> 0 and (amount > 0 or term >0);

drop table if exists analytics.SA_searchData_loans;
create table analytics.SA_searchData_loans as
  select cs_p_id, cs_ed, productname, interestrate, eligibility, amount, term, loantype, rank, event_date, prd_id
  from domo.prd_loan_event
  where cs_ed between to_date('01-01-19', 'dd-mm-yy') and to_date('31-03-19', 'dd-mm-yy') and (amount > 0 or term >0);

drop table if exists analytics.SA_populationData_loans;
create table analytics.SA_populationData_loans as
  select distinct cs_p_id, cs_ed, termrequested, amountrequested from domo.prd_ov_event
  where cs_ed between to_date('01-01-19', 'dd-mm-yy') and to_date('31-03-19', 'dd-mm-yy')
  and payloadtype='LOAN'
  and softsearch='true' and (searchreason isnull or searchreason='user');

drop table if exists analytics.SA_score_loans;
create table analytics.SA_score_loans as
select cs_p_id, cr_credit_score_i, event_date
from analytics.creditreports_full
where event_date between to_date('01-01-19', 'dd-mm-yy') and to_date('31-03-19', 'dd-mm-yy');

drop table if exists analytics.SA_fintab_loans;
create table analytics.SA_fintab_loans as
with t1 as (select a.*, b.amountrequested, b.termrequested from analytics.SA_searchData_loans as a
            left join analytics.SA_populationData_loans as b
            on a.cs_p_id=b.cs_p_id and a.cs_ed=b.cs_ed),
     t2 as (select t1.*, b.cr_credit_score_i from t1 left join
            analytics.SA_score_loans as b
            on t1.cs_p_id =b.cs_p_id
            and date_trunc('month', t1.event_date) = date_trunc('month', b.event_date)),
     t3 as (select *, Lead(cs_ed, 1) over(PARTITION BY cs_p_id, productname order by cs_ed) as Nxt_cs_ed
            from t2)
     select b.*, a.cs_ed as click_cs_ed, a.tm_product
     from t3 as b
     left join analytics.SA_clicksData_loans as a on
     b.cs_p_id=a.cs_p_id and b.productname=a.tm_product and ((a.cs_ed >= b.cs_ed and a.cs_ed < b.nxt_cs_ed) or (a.cs_ed >= b.cs_ed and b.nxt_cs_ed is NULL));

drop table if exists analytics.SA_fintab_2_loans;
create table analytics.SA_fintab_2_loans as
  select interestrate, eligibility, amount, term, loantype, rank, productname, event_date, termrequested, amountrequested,
  Case
    When tm_product isNull Then 0
    Else 1
  END as clicked,
  Case
    When cr_credit_score_i < 280 then 'VP'
    When cr_credit_score_i < 380 and cr_credit_score_i > 279 then 'P'
    When cr_credit_score_i < 420 and cr_credit_score_i > 379 then 'F'
    When cr_credit_score_i < 466 and cr_credit_score_i > 419 then 'G'
    ELSE 'E'
  END as score
  from analytics.SA_fintab_loans;

drop table if exists analytics.SA_loanmodel_data;
create table analytics.SA_loanmodel_data as
  Select a.*, b.issuer from analytics.SA_fintab_2_loans as a
  left join analytics.tj_flash_rev_mapping as b
  ON a.productname = b.productid;